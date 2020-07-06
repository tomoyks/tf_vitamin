import os
from slack import WebClient
from slack.errors import SlackApiError


class SlackNotifier:
    def __init__(self, channel_name='#kaggle-notifier', token=None):
        self.channel_name = channel_name

        token = self.load_token(token)
        self.client = WebClient(token=token)

    def load_token(self, token):
        if token is not None:
            return token

        if 'SLACK_API_TOKEN' in os.environ.keys():
            return os.environ.get['SLACK_API_TOKEN']

    def post_message(self, msg):
        try:
            self.client.chat_postMessage(
                channel=self.channel_name,
                text=msg)
        except SlackApiError as e:
            assert e.response['ok'] is False
            error = e.response['error']
            print(f'Got an error: {error}')

    def post_file(self, filename, title=None):
        try:
            self.client.files_upload(
                channels=self.channel_name,
                file=filename,
                title=title,
                filename=filename)
        except SlackApiError as e:
            assert e.response['ok'] is False
            error = e.response['error']
            print(f'Got an error: {error}')
