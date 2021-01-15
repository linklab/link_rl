from codes.a_config.b_atari_parameters.parameters_atari_breakout_dqn import PARAMETERS_BREAKOUT_DQN
from codes.a_config.a_basic_parameters.parameters_cartpole_a2c import PARAMETERS_CARTPOLE_A2C
from codes.a_config.a_basic_parameters.parameters_cartpole_dqn import PARAMETERS_CARTPOLE_DQN
from codes.a_config.d_pendulum_parameters.parameters_pendulum_ddpg import PARAMETERS_PENDULUM_DDPG

from codes.f_main.general_main.main_single import main

if __name__ == "__main__":
    print("\n################## {0} ##################".format(PARAMETERS_CARTPOLE_DQN.__name__))
    params = PARAMETERS_CARTPOLE_DQN
    params.MAX_GLOBAL_STEP = 2000
    main(params)

    print("\n################## {0} ##################".format(PARAMETERS_PENDULUM_DDPG.__name__))
    params = PARAMETERS_PENDULUM_DDPG
    params.MAX_GLOBAL_STEP = 2000
    main(params)

    print("\n################## {0} ##################".format(PARAMETERS_BREAKOUT_DQN.__name__))
    params = PARAMETERS_BREAKOUT_DQN
    params.MAX_GLOBAL_STEP = 2000
    main(params)

    print("\n################## {0} ##################".format(PARAMETERS_CARTPOLE_A2C.__name__))
    params = PARAMETERS_CARTPOLE_A2C
    params.MAX_GLOBAL_STEP = 2000
    main(params)
