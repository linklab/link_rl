### 1. TRAIN
- Enter the code below into ```e_main\config_single.py``` and run ```e_main\b_single_main_sequential.py``` 
  - if ```e_main\config_single.py``` does not exist, create it
  
- Train CartPole using DQN
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn
  config = ConfigCartPoleDqn()
  config.MAX_TRAINING_STEPS = 10_000
  config.USE_WANDB = False 
  ``` 
- Train CartPole using A2C
  ```python
  from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
  config = ConfigCartPoleA2c()
  config.MAX_TRAINING_STEPS = 50_000
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
  
### 2. PLAY
