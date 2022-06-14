import torch
print(torch.__version__) # 설치된 PyTorch 버전을 확인합니다. 1.12 이상이어야 합니다.
print(torch.backends.mps.is_built()) # MPS 장치를 지원하도록 빌드되어있는지 확인합니다. True여야 합니다.
print(torch.backends.mps.is_available()) # MPS 장치가 사용 가능한지 확인합니다. True여야 합니다.


import torch

mps_device = torch.device("mps")

# MPS 장치에 바로 tensor를 생성합니다.
x = torch.ones(5, device=mps_device)
# 또는
x = torch.ones(5, device="mps")

# GPU 상에서 연산을 진행합니다.
y = x * 2
