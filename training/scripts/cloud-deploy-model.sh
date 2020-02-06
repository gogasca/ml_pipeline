#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -x

REGION="us-central1" # choose a GCP region, e.g. "us-central1". Choose from https://cloud.google.com/ml-engine/docs/tensorflow/regions
BUCKET_NAME="news-ml" # change to your bucket name, e.g. "my-bucket"

# Model information
MODEL_NAME="sentiment_classifier"
MODEL_DIR=gs://"${BUCKET_NAME}"/models/
PACKAGES_DIR=gs://"${BUCKET_NAME}"/packages
MODEL_VERSION="v1" # change to your model version, e.g. "v1"

# Model Binaries corresponds to the tf.estimator.FinalExporter configuration in trainer/experiment.py
RUNTIME_VERSION=1.15
PYTHON_VERSION=3.5

PREDICTION_FILE="${PACKAGES_DIR}"/custom_prediction-0.1.tar.gz
TRAINER_FILE="${PACKAGES_DIR}"/trainer-0.1.tar.gz

# Deploy custom prediction + trainer
gsutil cp -R ./dist/*-0.1.tar.gz "${PACKAGES_DIR}"

# Delete model version, if previous model version exist.
gcloud ai-platform versions delete "${MODEL_VERSION}" --model="${MODEL_NAME}"

# Delete model, if previous model exist.
gcloud ai-platform models delete "${MODEL_NAME}"

# Deploy model to GCP
gcloud ai-platform models create "${MODEL_NAME}" --regions="${REGION}"

# Deploy model version
gcloud beta ai-platform versions create "${MODEL_VERSION}" \
 --model="${MODEL_NAME}" \
 --origin="${MODEL_DIR}" \
 --python-version="${PYTHON_VERSION}" \
 --runtime-version="${RUNTIME_VERSION}" \
 --package-uris="${PREDICTION_FILE}","${TRAINER_FILE}" \
 --prediction-class=model_prediction.CustomModelPrediction