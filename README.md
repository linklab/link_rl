# 1. TRAIN
- Enter the code below into ```e_main\config_single.py``` and run ```e_main\b_single_main_sequential.py``` 
  - if ```e_main\config_single.py``` does not exist, create it
  
- Train CartPole using DQN
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn
  config = ConfigCartPoleDqn()
  config.MAX_TRAINING_STEPS = 50_000
  config.USE_WANDB = False 
  ``` 
- Train CartPole using A2C
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
  config = ConfigCartPoleA2c()
  config.MAX_TRAINING_STEPS = 70_000
  config.USE_WANDB = False
  ```
- Train LunarLander using PPO
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpo
  config = ConfigLunarLanderPpo()
  config.MAX_TRAINING_STEPS = 100_000
  config.USE_WANDB = False  
  ```
- Train Pong using DQN
  ```python
  from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
  config = ConfigPongDqn()
  config.MAX_TRAINING_STEPS = 2_000_000
  config.USE_WANDB = False  
  ```
  
# 2. PLAY
- Enter the code below into ```e_main\config_single.py``` and run ```f_play\play.py``` 
  - if ```e_main\config_single.py``` does not exist, create it


- Complete to train the model and you will get the trained model file
  - ```f_play\models\{env_name}\{algorithm}\*.pth```
  - *.pth: ```{reward}_{std}_{year}_{month}_{day}_{env}_{algorithm}.pth```
  - e.g. 1
    - After successful train, you can see the trained model file ```500.0_0.0_2037_6_31_CartPole-v1_DQN.pth``` in the folder ```f_play\models\CartPole-v1\DQN```
    - And then, you need to add the following into the config_single.py
      - config.PLAY_MODEL_FILE_NAME = "500.0_0.0_2037_6_31_CartPole-v1_DQN.pth"
  - e.g. 2
    - After successful train, you can see the trained model file ```232.4_15.3_2019_8_19_LunarLander-v2_A2C.pth``` in the folder ```f_play\models\LunarLander-v2\A2C```
    - And then, you need to add the following into the config_single.py
     - config.PLAY_MODEL_FILE_NAME = "232.4_15.3_2019_8_19_LunarLander-v2_A2C.pth"

- Play CartPole using DQN
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn
  config = ConfigCartPoleDqn()
  config.PLAY_MODEL_FILE_NAME = [*.pth]
  config.USE_WANDB = False 
  ``` 
- Play CartPole using A2C
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
  config = ConfigCartPoleA2c()
  config.PLAY_MODEL_FILE_NAME = [*.pth]
  config.USE_WANDB = False
  ```
- Play LunarLander using PPO
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpo
  config = ConfigLunarLanderPpo()
  config.PLAY_MODEL_FILE_NAME = [*.pth]
  config.USE_WANDB = False  
  ```
- Play Pong using DQN
  ```python
  from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
  config = ConfigPongDqn()
  config.PLAY_MODEL_FILE_NAME = [*.pth]
  config.USE_WANDB = False  
  ```