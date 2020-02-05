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
# This scripts performs Cloud training for a TensorFlow model.
set -ev
echo "Training Cloud ML model"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="sentiment_$now"

MODEL_NAME="sentiment_classifier" # Change to your model name, e.g. "estimator"
RUNTIME_VERSION=1.15
PYTHON_VERSION=3.5

BUCKET_NAME="news-ml" # TODO Change to your bucket name.
REGION="us-central1"
PACKAGE_PATH=./trainer
MODEL_DIR="gs://$BUCKET_NAME/models"
TRAINING_FILE="gs://$BUCKET_NAME/datasets/training.csv"
GLOVE_FILE="gs://$BUCKET_NAME/datasets/glove.twitter.27B.50d.txt"
PACKAGES_DIR="gs://$BUCKET_NAME/packages"
SAVED_MODEL_NAME="keras_saved_model.h5"
PROCESSOR_STATE_FILE="processor_state.pkl"


gcloud ai-platform jobs submit training "${JOB_NAME}" \
        --stream-logs \
        --runtime-version="${RUNTIME_VERSION}" \
        --python-version="${PYTHON_VERSION}" \
        --module-name=trainer.task \
        --package-path="${PACKAGE_PATH}" \
        --region="${REGION}" \
        --config=./config.yaml \
        --job-dir="${MODEL_DIR}" \
        -- \
        --train-file="${TRAINING_FILE}" \
        --glove-file="${GLOVE_FILE}" \
        --learning-rate=0.001 \
        --embedding-dim=50 \
        --num-epochs=25 \
        --filter-size=64 \
        --batch-size=128 \
        --vocab-size=25000 \
        --pool-size=3 \
        --max-sequence-length=50 \
        --saved-model="${SAVED_MODEL_NAME}" \
        --preprocessor-state-file="${PROCESSOR_STATE_FILE}" \
        --deploy-gcp \
        --gcs-bucket="${BUCKET_NAME}" \


read -p "Press enter to continue"
rm -rf ../setup.py
# Recreate the Setup file
cat << 'EOF' > "../setup.py"
from setuptools import setup

setup(
  name="sentiment_classifier",
  version="0.1",
  include_package_data=True,
  scripts=["preprocess.py", "model_prediction.py"]
)
EOF

python setup.py sdist
gsutil cp ./dist/"${MODEL_NAME}"-0.1.tar.gz "${PACKAGES_DIR}"/"${MODEL_NAME}"-0.1.tar.gz

# Revert to original setup file
rm -rf ../setup.py
cat << 'EOF' > "../setup.py"
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs'
]

setup(
    name='trainer',
    description='AI Platform Training job for TensorFlow',
    author='Google Cloud Platform',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)
EOF