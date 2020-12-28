import csv
import glob
import os

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import DateTime, Column, Integer, Float, String, Unicode, Boolean, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import pandas as pd
import warnings
from sqlalchemy import exc as sa_exc

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=sa_exc.SAWarning)

from common.environments.trade.trade_constant import NAVER_MYSQL_ID, NAVER_MYSQL_PASSWORD, NAVER_MYSQL_HOST, TimeUnit, \
    CSV_RAW_DATA_FILE
from common.environments.trade.trade_utils import get_previous_one_unit_date_time

naver_ohlcv_price_engine = create_engine(
    'mysql+mysqlconnector://{0}:{1}@{2}/record?use_pure=True'.format(
        NAVER_MYSQL_ID, NAVER_MYSQL_PASSWORD, NAVER_MYSQL_HOST
    ),
    encoding='utf-8', echo=False
)

Base = declarative_base()

naver_ohlcv_price_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=naver_ohlcv_price_engine))


def get_candle_class(coin_name, time_unit):
   class Candle(Base):
        __tablename__ = "KRW_{0}_PRICE_{1}".format(coin_name, time_unit)
        __table_args__ = {'extend_existing': True}

        id = Column(Integer, primary_key=True, autoincrement=True)
        datetime_utc = Column(DateTime, unique=True, index=True)
        datetime_krw = Column(DateTime, unique=True, index=True)
        daily_base_timestamp = Column(Integer)

        open = Column(Float)
        high = Column(Float)
        low = Column(Float)
        final = Column(Float)
        volume = Column(Float)

        def __init__(self, *args, **kw):
            super(Candle, self).__init__(*args, **kw)

        def get_id(self):
            return self.id

        def __repr__(self):
            return str({
                "id": self.id,
                "datetime_krw": self.datetime_krw,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "final": self.final,
                "volume": self.volume
            })

   return Candle


def get_data(coin_name, time_unit):
    previous_one_unit_datetime_krw = get_previous_one_unit_date_time(time_unit)

    candle_target_coin_class = get_candle_class(coin_name, time_unit.value)
    candle_btc_coin_class = get_candle_class('BTC', time_unit.value)

    target_components = (coin_name, candle_target_coin_class)
    btc_components = ("BTC", candle_btc_coin_class)

    target_df = None
    btc_df = None

    for coin_name, candle_coin_class in [target_components, btc_components]:
        csv_raw_data_ok = False
        csv_raw_data_file = CSV_RAW_DATA_FILE.format(coin_name, time_unit.value, '*')

        assert len(glob.glob(csv_raw_data_file)) <= 1, "len(glob.glob(csv_raw_data_file)) is more that 1: {0}".format(csv_raw_data_file)

        if len(glob.glob(csv_raw_data_file)) == 1:
            exist_file_name = glob.glob(csv_raw_data_file)[0]

            if exist_file_name.endswith(previous_one_unit_datetime_krw.replace(" ", "T") + ".csv"):
                if coin_name == "BTC":
                    btc_df = pd.read_csv(exist_file_name)
                else:
                    target_df = pd.read_csv(exist_file_name)
                csv_raw_data_ok = True
            else:
                os.remove(exist_file_name)

        if not csv_raw_data_ok:
            queryset = naver_ohlcv_price_session.query(candle_coin_class).order_by(
                candle_coin_class.datetime_krw.asc()
            )
            df = pd.read_sql(queryset.statement, naver_ohlcv_price_session.bind)[:-1]
            last_datetime_krw = str(df.iloc[-1, 2])

            if last_datetime_krw != previous_one_unit_datetime_krw:
                df = df[:-1]

            print("[{0}] New Data From NAVER: {1}".format(coin_name, df.iloc[-1, :]['datetime_krw']))

            new_csv_raw_data_file = CSV_RAW_DATA_FILE.format(
                coin_name, time_unit.value, previous_one_unit_datetime_krw.replace(" ", "T")
            )

            df.to_csv(new_csv_raw_data_file, mode='w')

            if coin_name == "BTC":
                btc_df = pd.read_csv(new_csv_raw_data_file)
            else:
                target_df = pd.read_csv(new_csv_raw_data_file)

    data = pd.merge(
        left=target_df, right=btc_df, on='datetime_utc', how='inner', suffixes=('', '_btc')
    )

    state_data = data[[
        'daily_base_timestamp', 'open', 'high', 'low', 'final', 'volume',
         'final', 'open_btc', 'high_btc', 'low_btc', 'final_btc', 'volume_btc'
    ]].to_numpy()

    num_train_data = int(len(data) * 0.8)
    num_test_data = len(data) - num_train_data

    first_train_datetime_krw = str(data.iloc[0, 2])
    last_train_datetime_krw = str(data.iloc[num_train_data, 2])
    first_test_datetime_krw = str(data.iloc[num_train_data + 1, 2])
    last_test_datetime_krw = str(data.iloc[-1, 2])

    train_data = data[:num_train_data]
    train_state_data = state_data[:num_train_data]

    test_data = data[num_train_data:]
    test_state_data = state_data[num_train_data:]

    assert len(train_data) == len(train_state_data) == num_train_data
    assert len(test_data) == len(test_state_data) == num_test_data

    train_data_info = {
        "data": train_data,
        "state_data": train_state_data,
        "first_datetime_krw": first_train_datetime_krw,
        "last_datetime_krw": last_train_datetime_krw
    }

    test_data_info = {
        "data": test_data,
        "state_data": test_state_data,
        "first_datetime_krw": first_test_datetime_krw,
        "last_datetime_krw": last_test_datetime_krw
    }

    return train_data_info, test_data_info