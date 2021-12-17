### 1. 환경 만들기

- conda create -n link_rl python=3.8
- conda activate link_rl
- pytorch 설치
  - https://pytorch.org/ 참고
- OpenAI GYM
  - conda install -c conda-forge gym-all
- lz4
  - pip install lz4
- wandb
  - pip install wandb
- nvidia-ml-py3
  - pip install nvidia-ml-py3
- plotly
  - pip install plotly
- pandas
  - conda install pandas
  
### 2. gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"  

### 3. 새로운 패키지 설치 후 requirements.txt 새로 만들기

- pip freeze > requirements.txt

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