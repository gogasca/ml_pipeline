## Streaming Analytics

### Google Cloud Pub/Sub to AI Platform and then Google Cloud BigQuery

* [PubSubToBigQuery.py](PubSubToBigQuery.py)

The following example will run a streaming pipeline. It will read 
messages from a Pub/Sub topic, then window them into fixed-sized intervals, 
after that, it will call the Prediction API, once it gets the results,
will write them into BigQuery.

+ `--project`: sets the Google Cloud project ID to run the pipeline on
+ `--input-topic`: sets the input Pub/Sub topic to read messages from
+ `--region`: sets the region where Dataflow pipeline runs
+ `--staging-location`:  needed for executing the pipeline
+ `--temp-location`: needed for executing the pipeline
+ `--bigquery-dataset`:  BigQuery dataset reference
+ `--bigquery-table`: BigQuery table reference for output.
+ `--window-size [optional]`: specifies the window size in seconds, defaults to 60
+ `--min-batch-size [optional]`: specifies the batch min size
+ `--max-batch-size [optional]`: specifies the batch max size
+ `--runner`: specifies the runner to run the pipeline, if not set to `DataflowRunner`, `DirectRunner` is used

### Pre-requisites

### Streaming data

Run the [listener](../listener) example which gets Twitter streaming data
and pushes it to a PubSub topic.

### BigQuery 

1. Create a BigQuery dataset in BigQuery UI
2. Create BigQuery table using BigQuery UI

```
CREATE TABLE BIGQUERY_DATASET.BIGQUERY_TABLE (
   id STRING NOT NULL,
   text STRING,
   user_id STRING,
   sentiment FLOAT64,
   posted_at TIMESTAMP,
   favorite_count INT64,
   retweet_count INT64,
   media STRING
 )
 PARTITION BY DATE(_PARTITIONTIME)
 OPTIONS(
   description="A Twitter table"
 )
```

3. Define the DataFlow pipeline parameters
  
```
export INPUT_TOPIC=projects/news-ml-257304/topics/news-ml-twitter
export STAGING_LOCATION=gs://news-ml-dev/twitter-pipeline/staging
export TEMP_LOCATION=gs://news-ml-dev/twitter-pipeline/tmp
export REGION=us-central1
export BIGQUERY_DATASET=newsml_dataset_twitter
export BIGQUERY_TABLE=twitter_posts
export WINDOW_SIZE=60
export MIN_BATCH_SIZE=9
export MAX_BATCH_SIZE=10
export RUNNER=DataflowRunner
```

Define your Credentials file:
```
export GOOGLE_CLOUD_CREDENTIALS='key.json'
```
Execute the program as follows:

```
python PubSubToBigQuery.py \
    --input-topic=${INPUT_TOPIC} \
    --region=${REGION} \
    --staging-location=${STAGING_LOCATION} \
    --temp-location=${TEMP_LOCATION} \
    --bigquery-dataset=${BIGQUERY_DATASET} \
    --bigquery-table=${BIGQUERY_TABLE} \
    --window-size=${WINDOW_SIZE} \
    --min-batch-size=${MIN_BATCH_SIZE} \
    --max-batch-size=${MAX_BATCH_SIZE} \
    --runner=${RUNNER} &
```

