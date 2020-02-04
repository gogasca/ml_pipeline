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
"""Pre-process for Custom Prediction."""
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import re


class TextPreprocessor(object):
    def __init__(self, vocab_size, max_sequence_length):
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        self._tokenizer = None
        self._text_list_cleaned = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def text_list_cleaned(self):
        return self._text_list_cleaned

    @text_list_cleaned.setter
    def text_list_cleaned(self, text_list_cleaned):
        self._text_list_cleaned = text_list_cleaned

    def _clean_line(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#[A-Za-z0-9]+', '', text)
        text = text.replace('RT', '')
        text = text.lower()
        text = text.strip()
        return text

    def fit(self, text_list):
        # Create vocabulary from input corpus.
        _tokenizer = text.Tokenizer(num_words=self._vocab_size)
        _tokenizer.fit_on_texts(text_list)
        self.text_list_cleaned = [self._clean_line(txt) for txt in text_list]
        self.tokenizer = _tokenizer

    def transform(self, text_list):
        # Transform text to sequence of integers
        text_list = [self._clean_line(txt) for txt in text_list]
        text_sequence = self._tokenizer.texts_to_sequences(text_list)

        # Fix sequence length to max value. Sequences shorter than the length
        # are padded in the beginning and sequences longer are truncated
        # at the beginning.
        padded_text_sequence = sequence.pad_sequences(
            text_sequence, maxlen=self._max_sequence_length)
        return padded_text_sequence
