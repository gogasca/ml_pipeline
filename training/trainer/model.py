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

import tensorflow as tf

KERNEL_SIZES = [2, 5, 8]


def create_model(vocab_size, embedding_dim, filters, dropout_rate,
                 pool_size, embedding_matrix, max_sequence_length):
    """

    :param vocab_size:
    :param embedding_dim:
    :param filters:
    :param dropout_rate:
    :param pool_size:
    :param embedding_matrix:
    :param max_sequence_length:
    :return:
    """
    # Input layer
    model_input = tf.keras.layers.Input(shape=(max_sequence_length,),
                                        dtype='int32')
    # Embedding layer
    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        weights=[embedding_matrix]
    )(model_input)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    # Convolutional block
    conv_blocks = []
    for kernel_size in KERNEL_SIZES:
        conv = tf.keras.layers.Convolution1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            activation='relu',
            bias_initializer='random_uniform',
            strides=1)(layer)
        conv = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(conv)
        conv = tf.keras.layers.Flatten()(conv)
        conv_blocks.append(conv)

    layer = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 \
        else \
        conv_blocks[0]

    layer = tf.keras.layers.Dropout(dropout_rate)(layer)
    layer = tf.keras.layers.Dense(100, activation='relu')(layer)
    model_output = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
    model = tf.keras.models.Model(model_input, model_output)
    return model
