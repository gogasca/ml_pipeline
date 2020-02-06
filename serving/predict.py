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

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

PROJECT = ''
MODEL_NAME = ''
VERSION_NAME = ''

requests = [
    "god this episode sucks",
    "meh, I kinda like it",
    "what were the writer thinking, omg!",
    "omg! what a twist, who would'v though :o!",
    "woohoow, sansa for the win!"
]
# JSON format the requests
request_data = {'instances': requests}

api = discovery.build(
    'ml', 'v1',
    discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery'
                        '/ml_v1_discovery.json')

parent = 'projects/{}/models/{}/versions/{}'.format(PROJECT, MODEL_NAME,
                                                    VERSION_NAME)
parent = 'projects/{}/models/{}'.format(PROJECT, MODEL_NAME)
response = api.projects().predict(body=request_data, name=parent).execute()

print(response['predictions'])