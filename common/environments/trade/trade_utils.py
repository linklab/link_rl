import datetime as dt
import pytz
import numpy as np

from common.environments.trade.trade_constant import fmt, fmt2, TimeUnit


def convert_utc_to_seoul_time(date_time_utc, return_str=True):
    if isinstance(date_time_utc, type("")):
        date_time_utc = dt.datetime.strptime(date_time_utc, fmt)

    local_timezone = pytz.timezone('Asia/Seoul')
    local_time = date_time_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)

    if return_str:
        return local_time.strftime(fmt2)
    else:
        return local_time


def get_previous_one_unit_date_time(time_unit):
    date_time = convert_utc_to_seoul_time(dt.datetime.utcnow(), return_str=False)

    if time_unit == TimeUnit.ONE_HOUR:
        delta = dt.timedelta(hours=1)
    elif time_unit == TimeUnit.ONE_DAY:
        delta = dt.timedelta(days=1)
    else:
        raise ValueError()

    previous_date_time = date_time - delta
    date_time = previous_date_time.strftime(fmt2)

    if time_unit == TimeUnit.ONE_DAY:
        if date_time.endswith("00:00:00"):
            return date_time
        else:
            return date_time[:-8] + "00:00:00"
    elif time_unit == TimeUnit.ONE_HOUR:
        if date_time.endswith("00:00"):
            return date_time
        else:
            return date_time[:-5] + "00:00"
    else:
        raise ValueError("{0} unit is not supported".format(time_unit))


def get_current_unit_date_time(time_unit):
    date_time = convert_utc_to_seoul_time(dt.datetime.utcnow(), return_str=True)

    if time_unit == TimeUnit.ONE_DAY:
        if date_time.endswith("00:00:00"):
            return date_time
        else:
            return date_time[:-8] + "00:00:00"
    elif time_unit == TimeUnit.ONE_HOUR:
        if date_time.endswith("00:00"):
            return date_time
        else:
            return date_time[:-5] + "00:00"
    else:
        raise ValueError("{0} unit is not supported".format(time_unit))


def get_history_entry(data, previous_data):
    entry = [
        [data[0] - previous_data[0], data[6] - previous_data[6]],
        [data[1] - previous_data[1], data[7] - previous_data[7]],
        [data[2] - previous_data[2], data[8] - previous_data[8]],
        [data[3] - previous_data[3], data[9] - previous_data[9]],
        [data[4] - previous_data[4], data[10] - previous_data[10]],
        [data[5] - previous_data[5], data[11] - previous_data[11]],
    ]
    return np.asarray(entry)


def get_order_unit(price):
    if price >= 2000000:
        return 1000
    elif price >= 1000000:
        return 500 # ! 조심!!!!!
    elif price >= 500000:
        return 100
    elif price >= 100000:
        return 50
    elif price >= 10000:
        return 10
    elif price >= 1000:
        return 5
    elif price >= 100:
        return 1
    elif price >= 10:
        return 0.1
    else:
        return 0.01