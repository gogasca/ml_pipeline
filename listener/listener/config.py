"""Creates API object to contact Twitter streams"""
import tweepy
import logging
import os
from google.cloud import pubsub_v1

logger = logging.getLogger()


def get_authentication():
    """Authenticates in Twitter using environmental variables and creates a
    tweepy.API object.

    :return: tweepy.API object.
    """
    consumer_key = os.getenv('CONSUMER_KEY')
    consumer_secret = os.getenv('CONSUMER_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return auth


def get_topic(publisher):
    """

    :param publisher:
    :return:
    """
    project_id = os.getenv('PROJECT_ID', None)
    pubsub_topic = os.getenv('PUBSUB_TOPIC', None)
    topic_path = publisher.topic_path(project_id, pubsub_topic)
    return topic_path


def get_publisher():
    return pubsub_v1.PublisherClient()
