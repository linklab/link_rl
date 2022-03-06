# Installation for Linux

- Create environment
  ```commandline
  conda create -n link_rl_gym python==3.9
  ```

- Activate environment
  ```commandline
  conda activate link_rl_gym
  ```

- pytorch
  - go to https://pytorch.org/
  - check cuda
    ```commandline
    python -c "import torch; print(torch.cuda.is_available()); print(torch.rand(2,3).cuda());"
    ```

- Install package
  ```commandline
  pip install gym==0.22.0
  pip install pygame
  conda install -c conda-forge swig
  conda install -c conda-forge box2d-py
  pip install ale-py
  pip install gym[accept-rom-license]
  pip install opencv-python
  pip install lz4==4.0.0
  conda install -c fastai nvidia-ml-py3
  conda install -c plotly plotly
  conda install -c conda-forge wandb
  ```

- Test package(code)  
  Enter the code below into ```config_single.py``` and run ```b_single_main_sequential.py```
  - Test CartPole
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
    config = ConfigCartPoleA2c()
    config.MAX_TRAINING_STEPS = 1000
    ```
  - Test LunarLander
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpo
    config = ConfigLunarLanderPpo()
    config.MAX_TRAINING_STEPS = 1000
    ```
  - Test Pong
    ```python
    from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDqn
    config = ConfigPongDqn()
    config.MAX_TRAINING_STEPS = 1000
    ```
  


