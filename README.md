### 1. 환경 만들기
```commandline
conda create -n link_rl python=3.8
conda activate link_rl
```


### 2. 주요 패키지/모듈
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


### 3. gitignore 적용
```commandline
git rm -r --cached .
git add .
git commit -m "Apply .gitignore"  
```


### 4. Linux에 NFS 설치하고 MAC에서 원격 파일 시스템으로 MOUNT하기
- 참고
  - https://vitux.com/install-nfs-server-and-client-on-ubuntu/
  - https://jusungpark.tistory.com/36
- 1) Linux에서의 설정
  - sudo apt-get update
  - sudo apt install nfs-kernel-server
  - sudo chown nobody:nogroup /home/{account_name}/git
  - sudo chmod 777 /home/{account_name}/git 
  - sudo vi /etc/exports
    - /home/{account_name}/git 192.168.0.10(rw,sync,no_subtree_check)
      - *Your MAC IP: 192.168.0.10*
  - sudo exportfs -a
  - sudo systemctl restart nfs-kernel-server
  - sudo ufw allow from 192.168.0.0/24 to any port nfs
- 2) Mac에서의 설정
  - mkdir ~/linux_nfs_git
  - sudo mount -t nfs -o resvport,rw,nfc 192.168.0.43:/home/{account_name}/git ~/linux_nfs_git
      - *Your LINUX IP: 192.168.0.43*
    
  
### 5. Pytorch CUDA 사용 확인 
- python -c 'import torch; print(torch.rand(2,3).cuda())'
- nvidia-smi


### 6. database
- pip install sqlalchemy
- pip install mysql-connector


### 7. matlab
- cd "matlabroot/extern/engines/python"
- python setup.py install


### 8. grpc 
- python -m pip install --upgrade pip
- pip install grpcio
- pip install grpcio-tools
- stub 생성 방법 예: 
  - python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. rip_service.proto


### 9. swap 설치 및 에러 수정 
- https://kibua20.tistory.com/40
- http://aodis.egloos.com/5964233


### 10. Mujoco 설치 및 활용법기
- ubuntu
  - https://github.com/openai/mujoco-py에 접속하여 mujoco version 2.1 binaries를 다운받아 '~/.mujoco/mujoco210' 경로에 압축 해제
  - pip3 install free-mujoco-py 
  - sudo apt install libosmesa6-dev
  - sudo apt install patchelf
    **위의 두 패키지가 설치 안 될 경우**
    - sudo nano /etc/apt/sources.list
    - deb http://us.archive.ubuntu.com/ubuntu/ xenial main universe
    - deb-src http://us.archive.ubuntu.com/ubuntu/ xenial main universe
    - sources.list를 열어서 위의 두 라인 추가하여 다시 패키지 설치
  - b_environment/mujoco_gym_test.py 실행하여서 잘 설치 됐는지 확인  


### 11. Unity ML-Agent 설치 및 활용법


### 12. pynvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found
1. Copy C:\Windows\System32\nvml.dll
2. Past at C:\Program Files\NVIDIA Corporation\NVSMI
3. If not exist the directory, make the folder


### 13. Reference
- https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition
- NaN Handling: https://stable-baselines.readthedocs.io/en/master/guide/checking_nan.html