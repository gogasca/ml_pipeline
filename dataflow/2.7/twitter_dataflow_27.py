"""A streaming python pipeline to read in PubSub tweets and perform
classification"""

from __future__ import absolute_import

import argparse
import datetime
import json
import logging
import numpy as np
import os
import subprocess

import apache_beam as beam
import apache_beam.transforms.window as window

from apache_beam.io.gcp.bigquery_tools import parse_table_schema_from_json
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import PipelineOptions

from apache_beam.transforms.util import BatchElements

from googleapiclient import discovery


PROJECT_ID = os.getenv('PROJECT_ID')
PIPELINE_MODE = 'DataflowRunner'
DISCOVERY_SERVICE = 'https://storage.googleapis.com/cloud-ml/discovery' \
                    '/ml_v1_discovery.json'
BUCKET = os.getenv('BUCKET')
STAGING_LOCATION = os.getenv('STAGING_LOCATION')
TEMP_LOCATION = os.getenv('TEMP_LOCATION')
REGION = os.getenv('REGION')
DATASET = os.getenv('DATASET')
TWITTER_TABLE = os.getenv('TWITTER_TABLE')
TWITTER_TABLE_SENTIMENT = os.getenv('TWITTER_TABLE_SENTIMENT')
MODEL_URL = os.getenv('MODEL_URL')

api_client = None


def initialize_api():
    """

    :return:
    """
    global api_client
    if api_client is None:
        api_client = discovery.build('ml', 'v1',
                                     discoveryServiceUrl=DISCOVERY_SERVICE,
                                     cache_discovery=True)


def aggregate_format(key_values):
    # Aggregate tweets per 10 second window
    (key, values) = key_values

    mean_sentiment = np.mean([x['sentiment'] for x in values])
    mean_timestamp = datetime.datetime.utcfromtimestamp(np.mean([
        (datetime.datetime.strptime(x["posted_at"],
                                    '%Y-%m-%d %H:%M:%S') -
         datetime.datetime.fromtimestamp(
             0)).total_seconds() for x in values
    ]))

    logging.info("mean sentiment")
    logging.info(mean_sentiment)
    logging.info("mean timestamp")
    logging.info(mean_timestamp)

    # Return in correct format, according to BQ schema
    return {"posted_at": mean_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "sentiment": mean_sentiment}


def prediction(instances):
    """
    Calls the tweet_sentiment_classifier API on AI Platform to get predictions.

    Args:
        instances: list of strings.
    Returns:
        float: estimated values
    """

    # Init the Platform API
    initialize_api()
    logging.info("Making request to the ML API")
    values = []
    # Call the model
    try:
        response = api_client.projects().predict(body={'instances': instances},
                                                 name=MODEL_URL).execute()
        values = [item["score"] for item in response['predictions']]
        return values
    except Exception as e:
        return values


def estimate(messages):
    # Be able to cope with a single string as well
    if not isinstance(messages, list):
        messages = [messages]

    # Messages from PubSub are JSON strings
    instances = list(map(lambda message: json.loads(message), messages))

    # Estimate the sentiment of the 'text' of each tweet
    scores = prediction([instance["text"] for instance in instances])
    if scores:
        # Join them together
        for i, instance in enumerate(instances):
            instance['sentiment'] = scores[i]

        logging.info("First message in batch:")
        logging.info(instances[0])

        return instances
    else:
        logging.error('No scores')
        return


def run(argv=None):
    """Main pipeline run def.

    :param argv:
    :return:
    """

    # Make explicit BQ schema for output tables:
    # Tweets tables.
    bigquery_schema_json = {"fields": [{"name": "id", "type": "STRING"},
                                       {"name": "text", "type": "STRING"},
                                       {"name": "user_id", "type": "STRING"},
                                       {"name": "sentiment", "type": "FLOAT"},
                                       {"name": "posted_at",
                                        "type": "TIMESTAMP"}
                                       ]}
    bigquery_schema = parse_table_schema_from_json(
        json.dumps(bigquery_schema_json))
    """Tweets sentiment tables."""
    bigquery_schema_mean_json = {
        "fields": [{"name": "posted_at", "type": "TIMESTAMP"},
                   {"name": "sentiment", "type": "FLOAT"}]}
    bigquery_schema_mean = parse_table_schema_from_json(
        json.dumps(bigquery_schema_mean_json))

    """Build and run the pipeline."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--input_subscription',
        help=('Input PubSub subscription of the form '
              '"projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."'),
        default="projects/news-ml-257304/subscriptions/news-ml-twitter"
    )
    group.add_argument(
        '--input_topic',
        help=('Input PubSub topic of the form '
              '"projects/<PROJECT>/topics/<TOPIC>."'),
        default="projects/news-ml-257304/topics/news-ml-twitter"
    )

    known_args, pipeline_args = parser.parse_known_args(argv)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True

    # Run on Cloud DataFlow by default
    pipeline_options.view_as(StandardOptions).runner = PIPELINE_MODE
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT_ID
    google_cloud_options.staging_location = STAGING_LOCATION
    google_cloud_options.temp_location = TEMP_LOCATION
    google_cloud_options.region = REGION
    print(pipeline_options)
    p = beam.Pipeline(options=pipeline_options)

    # Read from PubSub into a PCollection.
    if known_args.input_subscription:
        lines = p | "read in tweets" >> beam.io.ReadFromPubSub(
            subscription=known_args.input_subscription,
            with_attributes=False,
            id_label="tweet_id"
        )
    else:
        lines = p | "read in tweets" >> beam.io.ReadFromPubSub(
            topic=known_args.input_topic,
            with_attributes=False,
            id_label="tweet_id")

    # Window them, and batch them into batches of 50 (not too large)
    output_tweets = (lines
                     | 'Assign window key' >> beam.WindowInto(
            window.FixedWindows(10))
                     | 'Batch into n batches' >> BatchElements(
            min_batch_size=49, max_batch_size=50)
                     | 'Predict sentiment' >> beam.FlatMap(
            lambda messages: estimate(messages))
                     )

    # Write to Bigquery
    output_tweets | 'store twitter posts' >> beam.io.WriteToBigQuery(
        table=TWITTER_TABLE,
        dataset=DATASET,
        schema=bigquery_schema,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        project=PROJECT_ID
    )

    # Average out and log the mean value
    (output_tweets
     | 'pair with key' >> beam.Map(lambda x: (1, x))
     | 'group by key' >> beam.GroupByKey()
     | 'aggregate and format' >> beam.Map(aggregate_format)
     | 'store aggregated sentiment' >> beam.io.WriteToBigQuery(
            table=TWITTER_TABLE_SENTIMENT,
            dataset=DATASET,
            schema=bigquery_schema_mean,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            project=PROJECT_ID))
    result = p.run()
    result.wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
