# AlphaGo Moment for Model Architecture Discovery

This is the official repository for our work "AlphaGo Moment for Model Architecture Discovery", an autonomous neural architecture search framework that uses Large Language Models (LLMs) to iteratively design, implement, and evaluate deep learning architectures through evolutionary algorithms and systematic experimentation in the domain of linear attention mechanisms.

## 📝 Introduction

Our framework represents a paradigm shift from traditional Neural Architecture Search (NAS) to autonomous AI research. Unlike conventional NAS methods that are limited to exploring human-defined search spaces, our framework enables AI to conduct genuine scientific discovery by autonomously hypothesizing novel architectural concepts, implementing them as code, and empirically validating their performance.

The system employs a multi-agent framework where specialized AI agents work collaboratively to evolve neural network architectures through systematic experimentation, performance analysis, and iterative improvement. Our framework has successfully discovered **106 novel, state-of-the-art linear attention architectures** through 1,773 autonomous experiments over 20,000 GPU hours, demonstrating that AI can autonomously generate world-class scientific knowledge.

![Pipeline Overview](images/new_pipeline.pdf)

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Performance](#-performance)
- [Get Started](#-get-started)
- [Acknowledgement](#-acknowledgement)
- [Citation](#️-citation)

## 🏆 Performance

Our framework has successfully demonstrated autonomous architecture discovery capabilities:

### Discovery Statistics
- **1,773 autonomous experiments** conducted over 20,000 GPU hours
- **106 novel architectures** achieving state-of-the-art performance
- **Linear scaling** relationship between compute budget and architecture discovery rate
- **Two-stage validation** from 20M to 340M parameter models

### Top-5 Discovered Architectures
Our final verification included 5 top-performing architectures trained on 15B tokens at 340M parameters:

| Architecture | Key Innovation | Performance Gain |
|-------------|----------------|------------------|
| **FusionGatedFIRNet** | Hierarchical two-stage routing with path-aware gating | Outperforms DeltaNet baseline |
| **ContentSharpRouter** | Content-aware gating with learnable temperature control | Superior benchmark performance |
| **PathGateFusionNet** | Parallel sigmoid fusion replacing softmax constraints | Enhanced context modeling |
| **HierGateNet** | Dynamic floors preventing pathway collapse | Improved long-range reasoning |
| **AdaMultiPathGateNet** | Unified token-level control with entropy regularization | Optimal path diversity |

### Evaluation Benchmarks
The system evaluates architectures across diverse cognitive domains:
- **Reasoning**: ARC Challenge/Easy, HellaSwag, Physical IQA, Social IQA, Winogrande
- **Language Understanding**: BoolQ, OpenBookQA, SQuAD
- **Memory & Context**: LAMBADA sequence modeling
- **Specialized Tasks**: FDA, SWDE structured data extraction

![Performance Analysis](images/combined_trend_analysis.pdf)

![Architecture Preferences](images/preference.pdf)

## 🚀 Get Started

### System Requirements

- Python 3.8+
- MongoDB 4.4+
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM, 32GB recommended

### Installation

```bash
git clone https://github.com/GAIR-NLP/AlphaGo-Moment-Architecture-Discovery.git
cd AlphaGo-Moment-Architecture-Discovery
conda create -n alphago-arch python=3.10
conda activate alphago-arch
pip install -r requirements.txt
```

### Core Dependencies

```bash
# Install PyTorch with CUDA support
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install additional requirements
pip install -r database/requirements.txt
```

### Environment Setup

1. **Configure Database**:
```bash
cd database
chmod +x start_api.sh
./start_api.sh
```

2. **Configure Framework**:
```bash
cd agent
# Edit config.py with your database URL
# DATABASE: "http://localhost:8000"
```

### Running Architecture Discovery

```bash
# Execute single evolution cycle
cd agent
python pipeline.py
```

The pipeline automatically:
1. Samples parent architecture from top-50 candidates using fitness-based selection
2. Generates novel architecture variation using Researcher module with cognition retrieval
3. Validates implementation and fixes errors through Engineer module
4. Trains and evaluates architecture across multiple benchmarks
5. Analyzes results and updates candidate pool through Analyst module

## 🙏 Acknowledgement

Our work builds upon foundational research in neural architecture search, linear attention mechanisms, and multi-agent AI systems. We acknowledge contributions from the research communities working on:

- Neural Architecture Search (NAS) and evolutionary algorithms
- Linear attention mechanisms and efficient Transformers
- Multi-agent AI frameworks and LLM-based code generation
- Automated machine learning and program synthesis

We especially thank the developers and maintainers of the key tools and frameworks that made this research possible:

- **FLAME Framework**: For providing the efficient training infrastructure and AdamW optimization with warmup-stabilize-decay learning rate schedules
- **LM-Evaluation-Harness**: EleutherAI's standardized evaluation framework that enabled consistent and reproducible benchmark assessments across all discovered architectures
- **Flash Linear Attention (FLA)**: For the foundational linear attention implementations and efficient computation kernels that supported our architecture explorations

## ✍️ Citation

Please cite this work if it contributes to your research:

```bibtex
@article{liu2025alphago_moment,
  title={AlphaGo Moment for Model Architecture Discovery},
  author={Liu, Yixiu and Nan, Yang and Xu, Weixian and Hu, Xiangkun and Qin, Zhen and Liu, Pengfei},
  journal={arXiv preprint},
  year={2025},
  note={Framework for autonomous neural architecture discovery in linear attention}
}
``` 