## Installation
Create environment
```commandline
conda create -n link_rl_gym python==3.9
```

Activate environment
```commandline
conda activate link_rl_gym
```

Install pytorch
- go to https://pytorch.org/ and install pytorch
- check cuda if you installed cuda-support version
  ```commandline
  python -c "import torch; print(torch.cuda.is_available()); print(torch.rand(2,3).cuda());"
  ```

Install package
- Linux 
  ```commandline
  pip install --upgrade pip
  sudo apt-get install swig
  pip install gym[nomujoco]
  pip install gym[atari,box2d,accept-rom-license]
  conda install -c conda-forge wandb
  conda install -c plotly plotly==5.6.0
  conda install -c conda-forge pandas
  conda install -c conda-forge matplotlib
  ```
- Windows
  ```commandline
  pip install gym==0.22.0
  pip install gym[atari,box2d,accept-rom-license]==0.22.0
  conda install -c conda-forge box2d-py==2.3.8
  pip install pygame==2.1.0
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  pip install plotly==5.6.0
  conda install -c conda-forge wandb
  conda install -c conda-forge pandas
  conda install -c conda-forge matplotlib
  ```
- Mac
  ```commandline
  pip install gym==0.22.0
  pip install gym[atari,box2d,accept-rom-license]==0.22.0
  pip install pygame==2.1.2
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  conda install -c plotly plotly==5.6.0
  conda install -c conda-forge wandb
  conda install -c conda-forge pandas
  conda install -c conda-forge matplotlib
  ```

  conda install -c conda-forge swig==4.0.2
  conda install -c conda-forge box2d-py==2.3.8
- 
## How to use link_rl
### Train
Enter the code below into ```e_main\config_single.py``` and run ```e_main\b_single_main_sequential.py``` 
- if ```e_main\config_single.py``` does not exist, create it
  
Train CartPole using DQN

```python
from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn

config = ConfigCartPoleDqn()
config.MAX_TRAINING_STEPS = 50_000
config.USE_WANDB = False 
``` 
Train CartPole using A2C

```python
from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA2c

config = ConfigCartPoleA2c()
config.MAX_TRAINING_STEPS = 70_000
config.USE_WANDB = False
```
Train LunarLander using PPO

```python
from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpo

config = ConfigLunarLanderPpo()
config.MAX_TRAINING_STEPS = 100_000
config.USE_WANDB = False  
```
Train Pong using DQN
```python
from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
config = ConfigPongDqn()
config.MAX_TRAINING_STEPS = 2_000_000
config.USE_WANDB = False  
```
 
### Play
Enter the code below into ```e_main\config_single.py``` and run ```f_play\play.py``` 
- if ```e_main\config_single.py``` does not exist, create it

Complete to train the model and you will get the trained model file
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

Play CartPole using DQN

```python
from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn

config = ConfigCartPoleDqn()
config.PLAY_MODEL_FILE_NAME = [*.pth]
config.USE_WANDB = False 
``` 
Play CartPole using A2C

```python
from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA2c

config = ConfigCartPoleA2c()
config.PLAY_MODEL_FILE_NAME = [*.pth]
config.USE_WANDB = False
```
Play LunarLander using PPO

```python
from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpo

config = ConfigLunarLanderPpo()
config.PLAY_MODEL_FILE_NAME = [*.pth]
config.USE_WANDB = False  
```
Play Pong using DQN
```python
from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
config = ConfigPongDqn()
config.PLAY_MODEL_FILE_NAME = [*.pth]
config.USE_WANDB = False  
```

#### MuJoCo210 or MuJoCo220
https://ropiens.tistory.com/178
https://github.com/openai/mujoco-py/issues/662

#### Gym-Robotics
pip install gym-robotics

***

#### To Do
- git action 만들기
- Learner 상속으로 if문 줄이기
- *.pth 파일명 yyyy_m_d에서 yyyy_mm_dd_hh_mm으로 변경
- *.pth에 weights만이 아니라 hyperparameters(config 인스턴스)도 함께 저장
- transition 전부 device에 올리고 transition과 같은 타입의 버퍼에 저장
- 학습할수록 속도가 느려지는 문제 해결
- recurrent 보수 작업 (recurrent_hidden)
- docstring 작성
- N_VECTORIZED_ENVS, N_ACTORS 늘릴 수 있도록 하기
- library와 main 분리
