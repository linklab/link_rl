import os
import configparser

PROJECT_HOME = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)

config = configparser.ConfigParser()
read_ok = config.read(os.path.join(PROJECT_HOME, "a_configuration", "a_config", "config.ini"))

SYSTEM_USER_NAME = config.get('SYSTEM', 'user_name', fallback="anonymous")
SYSTEM_COMPUTER_NAME = config.get('SYSTEM', 'computer_name', fallback="any_com")

SLACK_WEBHOOK_URL = config.get('SLACK', 'webhook_url', fallback='')
SLACK_API_TOKEN = config.get('SLACK', 'api_token', fallback='')
