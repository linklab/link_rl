### 1. 환경 만들기
```commandline
conda create -n link_rl python=3.9
conda activate link_rl
```


### 2. 주요 패키지/모듈
- 설치에 대한 최근 정보는 INSTALL_LINUX.md 와 INSTALL_WIN.md 참고

- pytorch 설치
  - https://pytorch.org/ 참고

- OpenAI GYM
  ```commandline
  conda install -c conda-forge gym-all
  conda install -c conda-forge gym-atari
  ```
- lz4
  ```commandline
  conda install -c conda-forge lz4
  ```
- wandb
  ```commandline
  conda install -c conda-forge wandb
  wandb login --relogin
  ```
- nvidia-ml-py3
  ```commandline
  conda install -c fastai nvidia-ml-py3
  ```
- plotly
  ```commandline
  conda install -c plotly plotly
  ```
- pandas
  ```commandline
  conda install -c conda-forge pandas
  ```
- matplotlib
  ```commandline
  conda install -c conda-forge matplotlib
  ```
- mpi4py
  ```commandline
  conda install -c conda-forge mpi4py
  ```
  - mpi4py 관련 MAC 에러 대처법 (Library not loaded: /usr/local/lib/libmpi.12.dylib)
    : https://stackoverflow.com/questions/35370396/mpi4py-library-not-loaded-usr-local-lib-libmpi-1-dylib
    : ln -s /usr/local/opt/open-mpi/lib/libmpi.40.dylib /usr/local/lib/libpmpi.12.dylib
  
- pybullet
  ```commandline
  conda install -c conda-forge pybullet
  ```
  - 설치 후 테스트
    ```commandline
    python -m pybullet_envs.examples.enjoy_TF_HumanoidBulletEnv_v0_2017may
    python -m pybullet_envs.examples.kukaGymEnvTest
    ```
  - 주의: pybullet-gym 설치하지 말것
  
- unity
  ```commandline
  pip install gym-unity
  ```

- mujoco_py
  ```commandline
  pip3 install -U 'mujoco-py<2.2,>=2.1'
  ```
  - https://github.com/openai/mujoco-py#install-mujoco
  - add following line to .bashrc:
     - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco210/bin
     - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
  - execute the followings in command window
  ```commandline
  sudo apt install clang
  sudo apt install libglew-dev
  sudo apt install patchelf
  ```



### Reference
- https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition
- NaN Handling: https://stable-baselines.readthedocs.io/en/master/guide/checking_nan.html
- https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
- https://github.com/MorvanZhou/PyTorch-Tutorial