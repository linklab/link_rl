from common.environments import TimeUnit, TradeEnvironmentType
from common.environments import UpbitEnvironment
from common.environments import get_previous_one_unit_date_time


def main():
    previous_one_datetime = get_previous_one_unit_date_time(TimeUnit.ONE_HOUR)
    print(previous_one_datetime)

    env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=TradeEnvironmentType.LIVE, params=params,
        previous_one_datetime=previous_one_datetime
    )

    state = env.reset()
    print(state.shape)


if __name__ == "__main__":
    main()