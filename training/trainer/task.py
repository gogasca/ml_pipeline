# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a Sentiment Analysis model using Twitter data from Kaggle.
We also use pre-compute Embeddings from Glove to provide additional information
to model in order to improve accuracy."""

import argparse
import logging
import os
import pickle
import subprocess
import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from . import preprocess
from . import model
from . import experiment


PROJECT = os.getenv('PROJECT_ID', 'news-ml-257304')
BUCKET = os.getenv('BUCKET', 'news-ml')
ROOT = 'ml_pipeline'
MODEL_DIR = os.path.join(ROOT, 'models').replace('\\','/')
PACKAGES_DIR = os.path.join(ROOT, 'packages').replace('\\','/')
TRAINING_FILE = 'training.csv'
GLOVE_FILE = 'glove.twitter.27B.50d.txt'

# Hyper-parameters.
CLASSES = {'negative': 0, 'positive': 1}  # label-to-int mapping
SENTIMENT_MAPPING = {
    0: 'negative',
    2: 'neutral',
    4: 'positive'
}
COLUMNS = {
    0: 'sentiment',
    1: 'id',
    2: 'posted_at',
    3: 'query',
    4: 'user_id',
    5: 'text'
}


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """

    args_parser = argparse.ArgumentParser()

    # Data files arguments
    args_parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data.',
        required=True)
    args_parser.add_argument(
        '--glove-file',
        help='Glove file from Stanford website.',
        default='glove.twitter.27B.50d.txt',
        required=True)
    args_parser.add_argument(
        '--eval-frequency-secs',
        help='How many seconds to wait before running the next evaluation.',
        default=15,
        type=int)
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        default=0.001,
        type=float)
    args_parser.add_argument(
        '--embedding-dim',
        help="""
        Embeddings Dimensions
        """,
        default=50,
        type=int,
    )
    args_parser.add_argument(
        '--filter-size',
        help="""
        Filters size
        """,
        default=64,
        type=int,
    )
    args_parser.add_argument(
        '--dropout-rate',
        help='Dropout rate.',
        default=0.5,
        type=float)
    args_parser.add_argument(
        '--pool-size',
        help="""
            Pool size
        """,
        default=3,
        type=int,
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""
            Maximum number of training data epochs on which to train.
            If both --train-size and --num-epochs are specified,
            --train-steps will default to:
                (train-size/train-batch-size) * num-epochs.
            """,
        default=25,
        type=int,
    )
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=128)
    args_parser.add_argument(
        '--train-size',
        help="""
        Size of the training data (instance count).
        If both --train-size and --num-epochs are specified,
        --train-steps will default to:
            (train-size/train-batch-size) * num-epochs.
        """,
        type=int,
        default=None)
    args_parser.add_argument(
        '--kernel-sizes',
        help="""
        Kernel sizes to use for DNN feature columns, provided in
        comma-separated layers.       
        """,
        nargs='+',
        type=int,
        default=[2, 5, 8])
    args_parser.add_argument(
        '--vocab-size',
        help="""
        Limit on the number vocabulary size used for tokenization.
        """,
        type=int,
        default=2500)
    args_parser.add_argument(
        '--max-sequence-length',
        help="""
        Sentences will be truncated/padded to this length.
        """,
        type=int,
        default=None)
    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=True)
    args_parser.add_argument(
        '--saved-model',
        help='Location to write Keras Saved Model in H5 format.',
        default='keras_saved_model.h5')
    args_parser.add_argument(
        '--preprocessor-state-file',
        help='GCS location to write checkpoints and export models.',
        default='./processor_state.pkl')
    args_parser.add_argument(
        '--deploy-gcp',
        action='store_true',
        default=False,
        help='Local or GCS location for writing checkpoints and exporting '
             'models')
    args_parser.add_argument(
        '--gcs-bucket',
        type=str,
        help='GCS bucket')
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""
        Flag to decide if the model checkpoint should be
        re-used from the job-dir.
        If set to False then the job-dir will be deleted.
        """)
    args_parser.add_argument(
        '--serving-export-format',
        help='The input format of the exported serving SavedModel.',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON')
    args_parser.add_argument(
        '--eval-export-format',
        help='The input format of the exported evaluating SavedModel.',
        choices=['CSV', 'EXAMPLE'],
        default='CSV')

    return args_parser.parse_args()


def _setup_logging():
    """Sets up logging."""
    root_logger = logging.getLogger()
    root_logger_previous_handlers = list(root_logger.handlers)
    for h in root_logger_previous_handlers:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    # Set tf logging to avoid duplicate logging. If the handlers are not removed
    # then we will have duplicate logging
    tf_logger = logging.getLogger('TensorFlow')
    while tf_logger.handlers:
        tf_logger.removeHandler(tf_logger.handlers[0])

    # Redirect INFO logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root_logger.addHandler(stdout_handler)

    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    

def copy_artifacts(source_path, destination_path):
    """
    :param source_path:
    :param destination_path:
    :return:
    """
    logging.info(
        'Moving model directory from {} to {}'.format(source_path,
                                                      destination_path))
    subprocess.call(
        "gsutil -m cp -r {} {}".format(source_path, destination_path),
        shell=True)


def get_coefficients(word, *arr):
    """

    :param word:
    :param arr:
    :return:
    """
    return word, np.asarray(arr, dtype='float32')


def get_embeddings(args, processor):
    """

    :param args:
    :param processor:
    :return:
    """
    if args.glove_file.startswith('gs://'):
        local_glove_file = os.path.basename(args.glove_file)
        copy_artifacts(args.glove_file, local_glove_file)
        args.glove_file = local_glove_file
    embeddings_index = dict(get_coefficients(*o.strip().split()) for o in
                            open(args.glove_file, 'r', encoding='utf8'))

    word_index = processor.tokenizer.word_index
    nb_words = min(args.vocab_size, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))

    for word, i in word_index.items():
        if i >= args.vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def main():
    args = get_args()
    _setup_logging()

    # If job_dir_reuse is False then remove the job_dir if it exists.
    logging.info('Resume training: {}'.format(args.reuse_job_dir))
    if not args.reuse_job_dir:
        if tf.io.gfile.exists(args.job_dir):
            tf.io.gfile.rmtree(args.job_dir)
            logging.info(
                'Deleted job_dir {} to avoid re-use'.format(args.job_dir))
    else:
        logging.info('Reusing job_dir {} if it exists'.format(args.job_dir))

    logging.info('Job directory: {}'.format(args.job_dir))
    logging.info('Keras saved model: {}'.format(args.saved_model))
    logging.info(
        'Pre-processor saved model: {}'.format(args.preprocessor_state_file))
    logging.info('Epoch count: {}.'.format(args.num_epochs))
    logging.info('Batch size: {}.'.format(args.batch_size))

    # Read Training file.
    df_twitter = pd.read_csv(args.train_file, encoding='latin1', header=None) \
        .rename(columns=COLUMNS)[['sentiment', 'text']]

    df_twitter['sentiment_label'] = df_twitter['sentiment'].map(
        SENTIMENT_MAPPING)

    sentiments = df_twitter.text
    labels = np.array(df_twitter.sentiment_label.map(CLASSES))

    # Train and test split
    x, _, y, _ = train_test_split(sentiments, labels, test_size=0.1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Create vocabulary from training corpus.
    processor = preprocess.TextPreprocessor(args.vocab_size,
                                            args.max_sequence_length)
    processor.fit(x_train)

    # Pre-process the data. (Transform text to sequence of integers)
    train_texts_vectorized = processor.transform(x_train)
    eval_texts_vectorized = processor.transform(x_test)

    # Creates a processor_state.pkl file.
    with open(args.preprocessor_state_file, 'wb') as f:
        pickle.dump(processor, f)

    embedding_matrix = get_embeddings(args, processor)

    # Create the Keras model.
    _model = model.create_model(args, embedding_matrix)

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    logging.info('Experiment started...')
    logging.info('.......................................')
    # Run experiment
    experiment.run(_model, args, train_texts_vectorized, y_train,
                   eval_texts_vectorized, y_test)

    time_end = datetime.utcnow()
    logging.info('.......................................')
    logging.info('Experiment finished.')
    time_elapsed = time_end - time_start
    logging.info('Experiment elapsed time: {} seconds'.format(
        time_elapsed.total_seconds()))

    if args.deploy_gcp:
        if not args.gcs_bucket:
            raise ValueError('No GCS bucket')
        # Copy Keras model
        model_gcs_path = os.path.join('gs://', args.gcs_bucket,
                                      args.saved_model)
        copy_artifacts(args.saved_model, model_gcs_path)
        # Copy Pre-processor
        process_gcs_path = os.path.join('gs://', args.gcs_bucket,
                                        args.preprocessor_state_file)
        copy_artifacts(args.preprocessor_state_file, process_gcs_path)


if __name__ == '__main__':
    main()
