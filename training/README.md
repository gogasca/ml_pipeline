# TensorFlow Sentiment Analysis

This notebook uses TensorFlow 1.15 to create a classification model for
classifying if text sentiment is positive or negative. It uses as
training data Kaggle Twitter information and also utilizes pre-compute
Embeddings to improve model results.

## Overview

The purpose of this directory is to provide a sample for how you can package a
TensorFlow training model to submit it to AI Platform. The sample makes it
easier to organise your code, and to adapt it to your dataset. In more details,
the template covers the following functionality:

*   Standard implementation of input, parsing, and serving functions.
*   Automatic feature columns creation based on the metadata (and normalization stats).
*   Train, evaluate, and export the model.
*   Parameterization of the experiment.

## Prerequisites
 
* Download the datasets from Kaggle and Stanford Glove website

This is the
[sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140/download).
It contains 1,600,000 tweets extracted using the Twitter API . The
tweets have been annotated (0 = negative, 4 = positive) and they can be
used to detect sentiment.

Download [pre-trained word
vectors](http://nlp.stanford.edu/data/glove.twitter.27B.zip) from
Stanford. Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download):

* Create a Python 3 virtual environment and activate it.
* Change the directory to this sample and run: 

```bash
  pip install -r requirements.txt
```

**Note:** These instructions are used for local testing. When you submit a training job, no code will be executed on 
your local machine.


## Sample Structure

* `trainer` directory: with all the python modules to adapt to your data
* `scripts` directory: command-line scripts to train the model locally or on AI Platform
* `requirements.txt`: containing all the required python packages for this sample 
* `config.yaml`: for hyper-parameter tuning and specifying the AI Platform scale-tier

### Trainer Template Modules

File Name                                         | Purpose                                                                                                                                                                                                                                                                                                                                | Do You Need to Change?
:------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------
[model.py](trainer/model.py)           | Includes: 1) function to create DNNLinearCombinedRegressor, and 2) DNNLinearCombinedClassifier.                                                                                                                                                                                                                                        | **No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc..
[experiment.py](trainer/experiment.py)       | Runs the model training and evaluation experiment, and exports the final model.                                                                                                                                                                                                                                                        | **No, unless** you want to add/remove parameters, or change parameter default values.
[task.py](trainer/task.py)             | Includes: 1) Initialise and parse task arguments (hyper parameters), and 2) Entry point to the trainer.                                                                                                                                                                                                                                | **No, unless** you want to add/remove parameters, or change parameter default values.
[model_prediction.py](trainer/model_prediction.py)             | Includes: 1) Read data and pre-process text before sending a Prediction request, and 2) Entry point to the AI Platform prediction.                                                                                                                                                                                                                                | **No, unless** you want to add/remove parameters, or change parameter default values.
[preprocess.py](trainer/preprocess.py)             | Includes: 1) Preprocess text data.                                                                                                                                                                                                                                | **No, unless** you want to add/remove parameters, or change parameter default values.


### Scripts

* [train-local.sh](scripts/train-local) This script trains a model
  locally. It generates a H5 model in local folder and verifies
  predictions locally.

* [train-cloud.sh](scripts/train-cloud.sh) This script submits a
  training job to AI Platform.

## How to run

Once the prerequisites are satisfied, you may:

1. For local training run:

```
source ./scripts/train-local.sh
```

2. For cloud training run:

```
source ./scripts/train-cloud.sh
```

### Custom prediction

We have created the sentiment-classifier package to be deployed in AI
Platform prediction, you can recreate it as instructed in Notebook or as
follows:

1. Update setup.py

```python
from setuptools import setup

setup(
  name="custom_prediction",
  version="0.1",
  include_package_data=True,
  scripts=["preprocess.py", "model_prediction.py"]
)
```
2. Copy `preprocess.py` and `model_prediction.py` to same folder as
`setup.py`
```

```
 
3. Install it

```shell script
 python setup.py sdist -formats=gztar
```
A new file under dist/ folder will be created.

4. Move it to GCS

```shell script 
gsutil cp ./dist/custom_prediction-0.1.tar.gz gs://"${PACKAGES_DIR}"/custom_prediction-0.1.tar.gz
```

### Versions

TensorFlow v1.14.0+

### References

https://www.kaggle.com/kazanova/sentiment140
https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
https://github.com/tensorflow/models/tree/master/research/sentiment_analysis
https://towardsdatascience.com/game-of-thrones-twitter-sentiment-with-keras-apache-beam-bigquery-and-pubsub-382a770f6583
https://towardsdatascience.com/keras-challenges-the-avengers-541346acb804