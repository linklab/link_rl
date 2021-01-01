import json
import requests
from slack import WebClient
from slack.errors import SlackApiError


class PushSlack:
    def __init__(self):
        if SLACK_WEBHOOK_URL_1 and SLACK_WEBHOOK_URL_1.startswith("http"):
            self.webhook_url_1 = SLACK_WEBHOOK_URL_1
        else:
            self.webhook_url_1 = None

        if SLACK_WEBHOOK_URL_2 and SLACK_WEBHOOK_URL_2.startswith("http"):
            self.webhook_url_2 = SLACK_WEBHOOK_URL_2
        else:
            self.webhook_url_2 = None

        self.slack_client = WebClient(token=SLACK_API_TOKEN)

    def send_message(self, username=None, message=None):
        slack_data = {'text': message}

        if self.webhook_url_1:
            requests.post(
                self.webhook_url_1,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )

        if self.webhook_url_2:
            requests.post(
                self.webhook_url_2,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )

    def send_message_to_manager(self, username=None, message=None):
        slack_data = {'text': message}

        if self.webhook_url_1:
            requests.post(
                self.webhook_url_1,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )

    def send_file_to_slack(self, filepath):
        try:
            response = self.slack_client.files_upload(
                channels='#bluebibi',
                file=filepath
            )
            assert response["file"]  # the uploaded file
        except SlackApiError as e:
            # You will get a SlackApiError if "ok" is False
            assert e.response["ok"] is False
            assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            print(f"Got an error: {e.response['error']}")