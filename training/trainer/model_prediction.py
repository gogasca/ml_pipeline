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
"""Use for AI Platform Custom Prediction."""

import os
import pickle
import numpy as np


class CustomModelPrediction(object):

    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def _postprocess(self, predictions):
        labels = ['negative', 'positive']
        return [
            {
                'label': labels[int(np.round(prediction))],
                'score': float(np.round(prediction, 4))
            } for prediction in predictions]

    def predict(self, instances, **kwargs):
        preprocessed_data = self._processor.transform(instances)
        predictions = self._model.predict(preprocessed_data)
        labels = self._postprocess(predictions)
        return labels

    @classmethod
    def from_path(cls, model_dir):
        import tensorflow.keras as keras
        model = keras.models.load_model(
            os.path.join(model_dir, 'keras_saved_model.h5'))
        with open(os.path.join(model_dir, 'processor_state.pkl'), 'rb') as f:
            processor = pickle.load(f)

        return cls(model, processor)
