"""A streaming python pipeline to read in PubSub tweets and perform
classification using Prediction API"""

import argparse
import datetime
import json
import logging
import numpy as np
import os
import socket
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
from googleapiclient.errors import HttpError

TIMEOUT_IN_SEC = 60 * 3 # 3 minutes timeout limit
socket.setdefaulttimeout(TIMEOUT_IN_SEC)

PROJECT_ID = os.getenv('PROJECT_ID')
DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery' \
                '/ml_v1_discovery.json'
MODEL_VERSION = 'projects/<YOUR-PROJECT-ID>/models/twitter'  # Required field.
api_client = None


def initialize_api():
    """

    :return:
    """
    global api_client
    if api_client is None:
        api_client = discovery.build('ml', 'v1',
                                     discoveryServiceUrl=DISCOVERY_URL,
                                     cache_discovery=True)


def prediction(instances):
    """Calls the Model prediction API on AI Platform to get scores.

    Args:
        instances: list of strings.
    Returns:
        [float]: A list of estimated values.
    """

    # Init the Platform API
    initialize_api()
    logging.info('Making request to the AI Platform API')
    # Call the model
    try:
        responses = api_client.projects().predict(body={'instances': instances},
                                                  name=MODEL_VERSION,
                                                  ).execute()
        return [response['score'] for response in responses['predictions']]
    except HttpError as err:
        logging.exception(err)


def prediction_helper(messages):
    """Processes PubSub messages and calls AI Platform prediction.

    :param messages:
    :return:
    """
    # Handle single string.
    if not isinstance(messages, list):
        messages = [messages]

    # Messages from PubSub are JSON strings
    instances = list(map(lambda message: json.loads(message), messages))

    # Estimate the sentiment of the 'text' of each tweet
    scores = prediction([instance['text'] for instance in instances])
    if scores:
        for i, instance in enumerate(instances):
            instance['sentiment'] = scores[i]
        logging.info('First message in batch: {}'.format(instances[0]))
        return instances
    else:
        logging.error('Invalid scores')
        return


def run(args, pipeline_args=None):
    """Executes Pipeline.

    :param args:
    :param pipeline_args:
    :return:
    """
    """Build and run the pipeline."""
    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(
        pipeline_args, streaming=True, save_main_session=True
    )
    pipeline_options.view_as(StandardOptions).runner = args.runner
    # Run on Cloud DataFlow by default
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT_ID
    google_cloud_options.staging_location = args.staging_location
    google_cloud_options.temp_location = args.temp_location
    google_cloud_options.region = args.region

    p = beam.Pipeline(options=pipeline_options)

    lines = p | 'read in tweets' >> beam.io.ReadFromPubSub(
        topic=args.input_topic,
        with_attributes=False,
        id_label='tweet_id')

    # Window them, and batch them into batches. (Not too large)
    output_tweets = (lines | 'assign window key' >> beam.WindowInto(
        window.FixedWindows(args.window_size))
                     | 'batch into n batches' >> BatchElements(
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size)
                     | 'predict sentiment' >> beam.FlatMap(
            lambda messages: prediction_helper(messages))
                     )

    # Make explicit BQ schema for output tables:
    bq_schema_json = {"fields": [{"name": "id", "type": "STRING"},
                                 {"name": "text", "type": "STRING"},
                                 {"name": "user_id", "type": "STRING"},
                                 {"name": "sentiment", "type": "FLOAT"},
                                 {"name": "posted_at", "type": "TIMESTAMP"},
                                 {"name": "favorite_count", "type": "INTEGER"},
                                 {"name": "retweet_count", "type": "INTEGER"},
                                 {"name": "media", "type": "STRING"},
                                 ]}
    bq_schema = parse_table_schema_from_json(json.dumps(bq_schema_json))

    # Write to BigQuery
    output_tweets | 'store twitter posts' >> beam.io.WriteToBigQuery(
        table=args.bigquery_table,
        dataset=args.bigquery_dataset,
        schema=bq_schema,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        project=PROJECT_ID
    )
    result = p.run()
    result.wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-topic',
        help='The Cloud Pub/Sub topic to read from.\n'
             'projects/<PROJECT_NAME>/topics/<TOPIC_NAME>'
    )
    parser.add_argument(
        '--region',
        help='The DataFlow region',
        default='us-central1'
    )
    parser.add_argument(
        '--staging-location',
        help='The DataFlow staging location',
        default='gs://<bucket_name>/staging/'
    )
    parser.add_argument(
        '--temp-location',
        help='The DataFlow temp location',
        default='gs://<bucket_name>/tmp/'
    )
    parser.add_argument(
        '--bigquery-dataset',
        help='BigQuery dataset',
        required=True
    )
    parser.add_argument(
        '--bigquery-table',
        help='BigQuery OutPut table',
        required=True
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=60,
        help="Output file's window size in number of seconds",
    )
    parser.add_argument(
        '--min-batch-size',
        type=int,
        default=9,
        help='Min batch size for Windowing',
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=10,
        help='Min batch size for Windowing',
    )
    parser.add_argument(
        '--runner',
        type=str,
        default='DataflowRunner',
        help='DataFlow running mode',
    )
    known_args, pipeline_args = parser.parse_known_args()

    run(
        known_args,
        pipeline_args
    )
