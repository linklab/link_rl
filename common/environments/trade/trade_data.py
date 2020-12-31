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
from common.environments.trade.trade_utils import get_previous_one_unit_date_time, get_current_unit_date_time, \
    convert_to_daily_timestamp

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
    current_unit_datetime_krw = get_current_unit_date_time(time_unit)
    current_unit_datetime_utc = get_current_unit_date_time(time_unit, krw=False)

    candle_target_coin_class = get_candle_class(coin_name, time_unit.value)
    candle_btc_coin_class = get_candle_class('BTC', time_unit.value)

    ### TARGET_COIN
    csv_target_raw_data_file = CSV_RAW_DATA_FILE.format(coin_name, time_unit.value, '*')

    assert len(glob.glob(csv_target_raw_data_file)) <= 1, \
        "len(glob.glob(csv_target_raw_data_file)) is more that 1: {0}".format(csv_target_raw_data_file)

    if len(glob.glob(csv_target_raw_data_file)) == 0:
        queryset = naver_ohlcv_price_session.query(candle_target_coin_class).order_by(
            candle_target_coin_class.datetime_krw.asc()
        )
        df = pd.read_sql(queryset.statement, naver_ohlcv_price_session.bind)
        last_stored_target_coin_datetime_krw = str(df.iloc[-1]["datetime_krw"])
        new_csv_raw_data_file = CSV_RAW_DATA_FILE.format(
            coin_name, time_unit.value, last_stored_target_coin_datetime_krw.replace(" ", "T").replace(":", "-")
        )
        df.to_csv(new_csv_raw_data_file, mode='w')

    file_name = glob.glob(csv_target_raw_data_file)[0]
    target_df = pd.read_csv(file_name)
    last_stored_target_coin_datetime_krw = str(target_df.iloc[-1]["datetime_krw"])

    print("[{0}] New Data From NAVER: {1}".format(coin_name, last_stored_target_coin_datetime_krw))

    if last_stored_target_coin_datetime_krw != current_unit_datetime_krw:
        assert last_stored_target_coin_datetime_krw == previous_one_unit_datetime_krw
        new_row = target_df.tail(1).copy(deep=True)
        new_row["id"] = new_row["id"] + 1
        new_row['datetime_utc'] = current_unit_datetime_utc
        new_row['datetime_krw'] = current_unit_datetime_krw
        new_row['daily_base_timestamp'] = convert_to_daily_timestamp(current_unit_datetime_krw)
        new_row['open'] = new_row['final']
        new_row['high'] = None
        new_row['low'] = None
        new_row['final'] = None
        new_row['volume'] = None
        target_df = target_df.append(new_row, ignore_index=True)

    ### BTC_COIN
    coin_name = "BTC"
    csv_btc_raw_data_file = CSV_RAW_DATA_FILE.format(coin_name, time_unit.value, '*')

    assert len(glob.glob(csv_btc_raw_data_file)) <= 1, \
        "len(glob.glob(csv_raw_data_file)) is more that 1: {0}".format(csv_btc_raw_data_file)

    if len(glob.glob(csv_btc_raw_data_file)) == 0:
        queryset = naver_ohlcv_price_session.query(candle_btc_coin_class).order_by(
            candle_btc_coin_class.datetime_krw.asc()
        )
        df = pd.read_sql(queryset.statement, naver_ohlcv_price_session.bind)
        last_stored_btc_coin_datetime_krw = str(df.iloc[-1]["datetime_krw"])
        new_csv_raw_data_file = CSV_RAW_DATA_FILE.format(
            coin_name, time_unit.value, last_stored_btc_coin_datetime_krw.replace(" ", "T").replace(":", "-")
        )
        df.to_csv(new_csv_raw_data_file, mode='w')

    file_name = glob.glob(csv_btc_raw_data_file)[0]
    btc_df = pd.read_csv(file_name)
    last_stored_btc_coin_datetime_krw = str(btc_df.iloc[-1]["datetime_krw"])

    print("[{0}] New Data From NAVER: {1}".format(coin_name, last_stored_btc_coin_datetime_krw))

    if last_stored_btc_coin_datetime_krw != current_unit_datetime_krw:
        assert last_stored_btc_coin_datetime_krw == previous_one_unit_datetime_krw
        new_row = btc_df.tail(1).copy(deep=True)
        new_row["id"] = new_row["id"] + 1
        new_row['datetime_utc'] = current_unit_datetime_utc
        new_row['datetime_krw'] = current_unit_datetime_krw
        new_row['daily_base_timestamp'] = convert_to_daily_timestamp(current_unit_datetime_krw)
        new_row['open'] = new_row['final']
        new_row['high'] = None
        new_row['low'] = None
        new_row['final'] = None
        new_row['volume'] = None
        btc_df = btc_df.append(new_row, ignore_index=True)

    data = pd.merge(
        left=target_df, right=btc_df, on='datetime_utc', how='inner', suffixes=('', '_btc')
    )

    state_data = data[[
        'daily_base_timestamp', 'open', 'high', 'low', 'final', 'volume',
        'final', 'open_btc', 'high_btc', 'low_btc', 'final_btc', 'volume_btc'
    ]].to_numpy()

    num_train_data = int(len(data) * 0.8)
    num_test_data = len(data) - num_train_data

    first_train_datetime_krw = str(data.iloc[0]['datetime_krw'])
    last_train_datetime_krw = str(data.iloc[num_train_data]['datetime_krw'])
    first_test_datetime_krw = str(data.iloc[num_train_data + 1]['datetime_krw'])
    last_test_datetime_krw = str(data.iloc[-1]['datetime_krw'])

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


if __name__ == "__main__":
    coin_name = "OMG"
    time_unit = TimeUnit.ONE_HOUR

    train_data_info, evaluate_data_info = get_data(coin_name=coin_name, time_unit=time_unit)

    print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
    print(evaluate_data_info["first_datetime_krw"], evaluate_data_info["last_datetime_krw"])
