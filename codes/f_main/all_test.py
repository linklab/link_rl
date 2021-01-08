from codes.a_config.fast_rl_parameters.parameters_fast_rl_breakout_dqn import PARAMETERS_FAST_RL_BREAKOUT_DQN
from codes.a_config.fast_rl_parameters.parameters_fast_rl_cartpole_dqn import PARAMETERS_FAST_RL_CARTPOLE_DQN
from codes.a_config.fast_rl_parameters.parameters_fast_rl_pendulum_ddpg import PARAMETERS_FAST_RL_PENDULUM_DDPG

from codes.f_main.general_main.main_single import main

if __name__ == "__main__":
    print("\n################## {0} ##################".format(PARAMETERS_FAST_RL_CARTPOLE_DQN.__name__))
    params = PARAMETERS_FAST_RL_CARTPOLE_DQN
    params.MAX_GLOBAL_STEP = 5000
    main(params)

    print("\n################## {0} ##################".format(PARAMETERS_FAST_RL_PENDULUM_DDPG.__name__))
    params = PARAMETERS_FAST_RL_PENDULUM_DDPG
    params.MAX_GLOBAL_STEP = 5000
    main(params)

    print("\n################## {0} ##################".format(PARAMETERS_FAST_RL_BREAKOUT_DQN.__name__))
    params = PARAMETERS_FAST_RL_BREAKOUT_DQN
    params.MAX_GLOBAL_STEP = 5000
    main(params)
