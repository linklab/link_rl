# BipedalWalker-v3 Hardcore

BipedalWalker-v3 Hardcore 환경을 이용한 강화학습 텀프로젝트입니다.

```bash
git clone https://github.com/dallu3123/link_BipedalWalker.git
cd link_BipedalWalker
```

## 환경 설치 (Mac)

### 1. Conda 환경 생성

```bash
conda create -n walker python=3.13 -y
conda activate walker
# 참고로 수업시간에서 생성한 환경에 그대로 이용해도 됩니다.
# conda activate link_rl

```

### 2. Swig 설치

Box2D 빌드에 필요한 swig를 먼저 설치합니다.

```bash
brew install swig
```

### 3. 패키지 설치

```bash
# Gymnasium + Box2D (BipedalWalker 환경)
pip install "gymnasium[box2d]"

# PyTorch
pip install torch

# 기타 유틸리티
pip install matplotlib numpy
```

### 4. 설치 확인

```bash
python -m test_walker
```

BipedalWalker 창이 뜨고 랜덤하게 움직이면 설치 완료입니다.

---

## 환경 설치 (Windows)

### 1. Conda 환경 생성

```bash
conda create -n walker python=3.13 -y
conda activate walker
# 참고로 수업시간에서 생성한 환경에 그대로 이용해도 됩니다.
# conda activate link_rl
```

### 2. Swig 설치

```bash
conda install -c anaconda swig
```

### 3. Microsoft C++ Build Tools 설치

Box2D 빌드에 필요합니다.
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 에서 설치 후
**"C++ build tools"** 체크 후 설치

### 4. 패키지 설치

```bash
# Gymnasium + Box2D (BipedalWalker 환경)
pip install "gymnasium[box2d]"

# PyTorch (CUDA 버전은 https://pytorch.org 에서 확인)
pip install torch

# 기타 유틸리티
pip install matplotlib numpy
```

### 5. 설치 확인

```bash
python -m test_walker
```

BipedalWalker 창이 뜨고 랜덤하게 움직이면 설치 완료입니다.

---

## 환경 설치 (Linux)

### 1. Conda 환경 생성

```bash
conda create -n walker python=3.13 -y
conda activate walker
# 참고로 수업시간에서 생성한 환경에 그대로 이용해도 됩니다.
# conda activate link_rl
```

### 2. 시스템 패키지 설치

```bash
sudo apt-get update
sudo apt-get install -y swig build-essential
```

### 3. 패키지 설치

```bash
# Gymnasium + Box2D (BipedalWalker 환경)
pip install "gymnasium[box2d]"

# PyTorch (CUDA 버전은 https://pytorch.org 에서 확인)
pip install torch

# 기타 유틸리티
pip install matplotlib numpy
```

### 4. 설치 확인

```bash
python -m test_walker
```

BipedalWalker 창이 뜨고 랜덤하게 움직이면 설치 완료입니다.
