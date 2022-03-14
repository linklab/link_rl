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
  pip install gym==0.22.0
  pip install pygame==2.1.2
  conda install -c conda-forge swig==4.0.2
  conda install -c conda-forge box2d-py==2.3.8
  pip install ale-py==0.7.4
  pip install gym[accept-rom-license]==0.22.0
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  conda install -c plotly plotly==5.6.0
  conda install -c conda-forge wandb
  conda install -c conda-forge pandas
  conda install -c conda-forge matplotlib
  ```
