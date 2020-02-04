#!/usr/bin/env python
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
"""Configures RunConfig."""

import logging
import math
import os
import tensorflow as tf


def create_run_config(args):
    """Create a tf.estimator.RunConfig object.

    Args:
      args: experiment parameters.
    """

    # Configure the distribution strategy if GPUs available.
    distribution_strategy = None
    # Get the available GPU devices
    num_gpus = len([device_name
                    for device_name in tf.contrib.eager.list_devices()
                    if "/device:GPU" in device_name])
    logging.info("%s GPUs are available.", str(num_gpus))
    if num_gpus > 1:
        distribution_strategy = tf.distribute.MirroredStrategy()
        logging.info("MirroredStrategy will be used for training.")
        # Update the batch size
        args.batch_size = int(math.ceil(args.batch_size / num_gpus))

    # Create RunConfig
    return tf.estimator.RunConfig(
        tf_random_seed=19831006,
        log_step_count_steps=100,
        model_dir=args.job_dir,
        save_checkpoints_secs=args.eval_frequency_secs,
        keep_checkpoint_max=3,
        train_distribute=distribution_strategy,
        eval_distribute=distribution_strategy)


def run(model, args, train_texts_vectorized, y_train, eval_texts_vectorized,
        y_test):
    """

    :param model:
    :param args:
    :param train_texts_vectorized:
    :param y_train:
    :param eval_texts_vectorized:
    :param y_test:
    :return:
    """
    optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(
        train_texts_vectorized,
        y_train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(eval_texts_vectorized, y_test),
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                min_delta=0.005,
                patience=3,
                factor=0.5),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.005,
                patience=5,
                verbose=0,
                mode='auto'
            ),
            tf.keras.callbacks.History()
        ]
    )
    with open('history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    model.save(args.saved_model)
