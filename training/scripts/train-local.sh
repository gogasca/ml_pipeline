#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs local training for a TensorFlow model.
set -ev
echo "Training local ML model"

MODEL_NAME="sentiment_classifier" # Change to your model name, e.g. "estimator"

PACKAGE_PATH=./trainer
MODEL_DIR="models"
TRAINING_FILE="training.csv"
GLOVE_FILE="glove.twitter.27B.50d.txt"
PACKAGES_DIR="packages"
BUCKET_NAME="" # TODO Change to your bucket name.
SAVED_MODEL_NAME="keras_saved_model.h5"
PROCESSOR_STATE_FILE="processor_state.pkl"


gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train-file=${TRAINING_FILE} \
        --glove-file=${GLOVE_FILE} \
        --learning-rate=0.001 \
        --embedding-dim=50 \
        --num-epochs=25 \
        --filter-size=64 \
        --batch-size=128 \
        --vocab-size=25000 \
        --pool-size=3 \
        --max-sequence-length=50 \
        --job-dir=${MODEL_DIR} \
        --saved-model=${SAVED_MODEL_NAME} \
        --preprocessor-state-file=${PROCESSOR_STATE_FILE}



EOF
python setup.py sdist
gsutil cp ./dist/${MODEL_NAME}-0.1.tar.gz gs://${BUCKET_NAME}/${PACKAGES_DIR}/${MODEL_NAME}-0.1.tar.gz
gsutil cp ${SAVED_MODEL_NAME} gs://${BUCKET_NAME}/${MODEL_DIR}/
gsutil cp ${PROCESSOR_STATE_FILE}.pkl gs://${BUCKET_NAME}/${MODEL_DIR}/

