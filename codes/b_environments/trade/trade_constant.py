import configparser
import enum
import os

from codes.e_utils.names import PROJECT_HOME

config_parser = configparser.ConfigParser()
read_ok = config_parser.read(os.path.join(PROJECT_HOME, "config", "codes.a_config.ini"))

fmt = "%Y-%m-%dT%H:%M:%S"
fmt2 = "%Y-%m-%d %H:%M:%S"

CSV_RAW_DATA_FNAMES = [
    'id',
    'datetime_utc',
    'datetime_krw',
    'daily_base_timestamp',
    'open', 'high', 'low', 'final', 'volume'
]


class TimeUnit(enum.Enum):
    ONE_HOUR = '1_HOUR'
    ONE_DAY = '1_DAY'


class Action(enum.Enum):
    HOLD = 0
    MARKET_BUY = 1
    MARKET_SELL = 2


class EnvironmentType(enum.Enum):
    TRAIN = 0
    TEST_RANDOM = 1
    TEST_SEQUENTIAL = 2
    LIVE = 3

OHLCV_FEATURES = ["daily_base_timestamp", "open", "high", "low", "final", "volume"]
SIZE_OF_OHLCV_FEATURE = len(OHLCV_FEATURES)

NAVER_MYSQL_ID = config_parser['MYSQL_NAVER']['mysql_id']
NAVER_MYSQL_PASSWORD = config_parser['MYSQL_NAVER']['mysql_password']
NAVER_MYSQL_HOST = config_parser['MYSQL_NAVER']['mysql_host']

SLACK_WEBHOOK_URL_1 = config_parser['SLACK']['webhook_url_1']
SLACK_WEBHOOK_URL_2 = config_parser['SLACK']['webhook_url_2']
SLACK_API_TOKEN = config_parser['SLACK']['SLACK_API_TOKEN']

RAW_DATA_DIR = os.path.join(PROJECT_HOME, "common", "environments", "trade", "raw_data")
CSV_RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, '{0}_{1}_{2}.csv')

if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)

TIME_UNIT = TimeUnit.ONE_HOUR
WINDOW_SIZE = 36
MAX_BUY_SIZE = 10
INITIAL_TOTAL_KRW = 1000000
BUY_AMOUNT = 100000
COMMISSION_RATE = 0.0005
SLIPPAGE_COUNT = 1
SLIPPAGE_RATE = 0.01