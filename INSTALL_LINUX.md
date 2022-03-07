# Installation for Linux

- Create environment
  ```commandline
  conda create -n link_rl_gym python==3.9
  ```

- Activate environment
  ```commandline
  conda activate link_rl_gym
  ```

- Install pytorch
  - go to https://pytorch.org/ and install pytorch
  - check cuda
    ```commandline
    python -c "import torch; print(torch.cuda.is_available()); print(torch.rand(2,3).cuda());"
    ```

- Install package
  ```commandline
  pip install gym==0.22.0
  pip install pygame==2.1.2
  conda install -c conda-forge swig==4.0.2
  conda install -c conda-forge box2d-py==2.3.8
  pip install ale-py==0.7.4
  pip install gym[accept-rom-license]==0.4.2
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  conda install -c fastai nvidia-ml-py3==7.352.0
  conda install -c plotly plotly==5.6.0
  conda install -c conda-forge wandb
  ```

- Test package(code)  
  Enter the code below into ```e_main\config_single.py``` and run ```e_main\b_single_main_sequential.py```  
  (if ```e_main\config_single.py``` does not exist, create it)
  - Train CartPole using A2C
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
    config = ConfigCartPoleA2c()
    config.MAX_TRAINING_STEPS = 1000
    ```
  - Train LunarLander using PPO
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpo
    config = ConfigLunarLanderPpo()
    config.MAX_TRAINING_STEPS = 1000
    ```
  - Train Pong using DQN
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDqn
    config = ConfigPongDqn()
    config.MAX_TRAINING_STEPS = 1000
    ```
