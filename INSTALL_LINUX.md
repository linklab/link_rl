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
  - check cuda if you installed cuda-support version
    ```commandline
    python -c "import torch; print(torch.cuda.is_available()); print(torch.rand(2,3).cuda());"
    ```

- Install package
 
  ```commandline
  pip install --upgrade pip
  sudo apt-get install swig
  pip install gym[nomujoco]
  pip install gym[atari]
  conda install -c conda-forge wandb
  conda install -c plotly plotly==5.6.0
  ```