# Removed Dependencies - Phase 1 Cleanup

## Summary
- **Original dependencies**: 150+ packages
- **New core dependencies**: ~60 packages
- **Reduction**: ~90 packages (60% reduction)

## Removed Categories

### 1. TensorFlow Ecosystem (Completely Unused)
```
tensorflow==2.19.0
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-probability==0.25.0
keras==3.10.0
tf_keras==2.19.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
```

### 2. Google/TensorFlow Support Libraries
```
google-pasta==0.2.0
gast==0.6.0
astunparse==1.6.3
flatbuffers==25.2.10
libclang==18.1.1
ml_dtypes==0.5.1
opt_einsum==3.4.0
```

### 3. Unused Development/Deployment Tools
```
ray==2.46.0
wandb==0.20.1
gym==0.26.2
gym-notices==0.0.8
griffe==1.7.3
GitPython==3.1.44
gitdb==4.0.12
smmap==5.0.2
```

### 4. Data Science Libraries (Moved to Optional)
```
matplotlib==3.10.3
seaborn==0.13.2
pandas==2.3.0
pyarrow==20.0.0
scipy==1.15.3
contourpy==1.3.2
cycler==0.12.1
fonttools==4.58.2
```

### 5. CUDA Dependencies (Will be replaced in Phase 2)
```
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
triton==3.1.0
```

### 6. PyTorch Lightning and Training Libraries (Moved to Optional)
```
pytorch-lightning==2.5.1.post0
lightning-utilities==0.14.3
torchmetrics==1.7.2
```

### 7. Redundant HTTP/Async Libraries
```
aiohappyeyeballs==2.6.1
aiosignal==1.3.2
httpcore==1.0.9
httpx-sse==0.4.0
multidict==6.4.4
yarl==1.20.0
frozenlist==1.6.2
aiohttp==3.12.9
propcache==0.3.1
```

### 8. Misc Unused Libraries
```
absl-py==2.3.0
attrs==25.3.0
click==8.2.1
cloudpickle==3.1.1
decorator==5.2.1
dill==0.3.8
distro==1.9.0
dm-tree==0.1.9
h5py==3.14.0
hf-xet==1.1.3
jiter==0.10.0
mcp==1.9.3
msgpack==1.1.0
multiprocess==0.70.16
namex==0.1.0
networkx==3.5
ninja==1.11.1.4
optree==0.16.0
ruamel.yaml==0.18.13
ruamel.yaml.clib==0.2.12
sentry-sdk==2.29.1
six==1.17.0
sse-starlette==2.3.6
termcolor==3.1.0
types-requests==2.32.0.20250602
typing-inspection==0.4.1
tzdata==2025.2
Werkzeug==3.1.3
wrapt==1.17.2
xxhash==3.5.0
```

## New Structure

### Core Requirements (requirements.txt)
Essential packages needed for basic ASI-Arch functionality (~60 packages)

### Optional Groups
- **requirements-data.txt**: Data analysis and visualization
- **requirements-dev.txt**: Development tools and testing
- **requirements-training.txt**: Advanced training and monitoring

### Component-Specific
- **database/requirements.txt**: Minimized for database service
- **cognition_base/requirements.txt**: Minimized for RAG service

## Installation Instructions

### Minimal Installation
```bash
pip install -r requirements.txt
pip install -r database/requirements.txt
pip install -r cognition_base/requirements.txt
```

### With Optional Components
```bash
# Add data science capabilities
pip install -r requirements-data.txt

# Add development tools
pip install -r requirements-dev.txt

# Add advanced training features
pip install -r requirements-training.txt
```

## Impact
- **Faster installation**: ~60% reduction in package count
- **Smaller environments**: Reduced disk space and memory usage
- **Fewer conflicts**: Less chance of dependency version conflicts
- **Better maintainability**: Easier to update and manage core dependencies