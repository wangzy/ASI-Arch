# ASI-Arch: Autonomous Neural Architecture Search through AI Agents

This is the official repository for ASI-Arch, an autonomous neural architecture search framework that uses AI agents to iteratively design, implement, and evaluate deep learning architectures through evolutionary algorithms and systematic experimentation.

## 📝 Introduction

ASI-Arch is a comprehensive framework for automated neural architecture discovery and optimization. The system employs multiple specialized AI agents working in concert to evolve neural network architectures through systematic experimentation, performance analysis, and iterative improvement. Our framework demonstrates emergent **architectural innovation behaviors**, including the ability to identify performance bottlenecks, synthesize research insights, implement novel computational mechanisms, and maintain theoretical rigor while achieving practical improvements.

The system uses UCT (Upper Confidence Trees) algorithm for intelligent candidate sampling, RAG (Retrieval-Augmented Generation) for research insight retrieval, MongoDB for experimental data management, and sophisticated agent orchestration for autonomous architecture evolution.

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Architecture](#-architecture)
- [Components](#-components)
- [Performance](#-performance)
- [Get Started](#-get-started)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Acknowledgement](#-acknowledgement)
- [Citation](#️-citation)

## 🏗️ Architecture

ASI-Arch implements an autonomous neural architecture search loop with the following key components:

### Core Evolutionary Loop
1. **Program Sampling**: Uses UCT algorithm to select promising parent architectures from top-performing candidates
2. **Architecture Evolution**: AI agents generate novel architecture variations based on experimental context and research insights
3. **Training & Evaluation**: Automated training with debugging and performance assessment across multiple benchmarks
4. **Analysis & Storage**: Comprehensive performance analysis with results stored for future evolution cycles

### Core Agent System
- **Creator/Optimizer Agents**: Design novel neural architectures based on experimental evidence and research insights
- **Code Checker Agent**: Validates and fixes implementation issues while preserving architectural innovations
- **Trainer Agent**: Executes training scripts with automatic debugging and retry mechanisms
- **Analyzer Agent**: Conducts comprehensive performance analysis and identifies architectural bottlenecks
- **Synthesizer Agent**: Extracts insights from experimental results to guide future iterations

### Supporting Infrastructure
- **UCT Candidate Manager**: Maintains top-50 performing architectures with intelligent sampling for exploration/exploitation
- **RAG Service**: Provides research insight retrieval for architecture innovation based on cognitive patterns
- **MongoDB Database**: Stores experimental results, code implementations, and performance metrics with FAISS indexing
- **Agent Tools**: Specialized utilities for code manipulation, file operations, and system interaction

```
┌─────────────────┐    UCT Sample  ┌─────────────────┐
│   Candidate     │ ──────────────► │   Evolution     │
│   Database      │                 │   Agents        │
│   (Top-50)      │                 │   + RAG         │
└─────────────────┘                 └─────────────────┘
         ▲                                   │
         │                                   ▼
    Store Results                   ┌─────────────────┐
    & Update                        │   Architecture  │
    Rankings                        │   Implementation│
         │                          │   + Validation  │
         │                          └─────────────────┘
┌─────────────────┐    Training     ┌─────────────────┐
│   Analysis &    │ ◄──────────────│   Automated     │
│   Synthesis     │    Results      │   Training &    │
│   Agents        │                 │   Evaluation    │
└─────────────────┘                 └─────────────────┘
```

## 🧩 Components

### Agent System (`/agent/`)
- **Pipeline Engine**: Orchestrates the complete experiment lifecycle
- **Evolution Module**: Implements architecture design and optimization strategies
- **Evaluation Framework**: Handles training execution and performance assessment
- **Analysis Engine**: Provides comprehensive experimental result analysis
- **Database Interface**: Manages experimental data storage and retrieval

### RAG Service (`/rag/`)
- **OpenSearch Integration**: Vector-based research insight retrieval
- **Cognitive Database**: Stores experimental trigger patterns and research findings
- **REST API**: HTTP interface for research insight queries
- **Docker Support**: Containerized deployment for scalability

### Database System (`/database/`)
- **MongoDB Backend**: Persistent storage for experimental data
- **FAISS Integration**: Efficient similarity search for architecture patterns
- **Candidate Management**: Tracks and manages experimental candidates
- **Evaluation Metrics**: Comprehensive performance tracking and analysis

## 🏆 Performance

ASI-Arch demonstrates significant improvements in automated neural architecture search:

- **Autonomous Evolution**: Generates novel architectures without human intervention through AI agent collaboration
- **UCT-Guided Search**: Efficiently explores the architecture space using Upper Confidence Trees algorithm
- **Performance Optimization**: Achieves consistent improvements over baseline architectures through systematic experimentation
- **Theoretical Grounding**: Maintains mathematical rigor while exploring innovative designs guided by research insights
- **Efficiency Preservation**: Ensures sub-quadratic complexity in all generated architectures through automatic validation
- **Robust Training**: Automated debugging and retry mechanisms ensure reliable evaluation across diverse architectures

### Architecture Evolution Results
The framework has successfully evolved various DeltaNet variants with measurable improvements across cognitive benchmarks:

| Architecture Variant | Train Loss ↓ | Test Performance ↑ | Key Innovation |
|---------------------|--------------|-------------------|----------------|
| DeltaNet (Baseline) | 4.5749 | 0.3623 | Base Architecture |
| Gated DeltaNet | 4.3772 | 0.3660 | Adaptive Gating Mechanisms |
| AdaptiveEntropyRouter | 4.4018 | 0.3747 | Dynamic Routing Systems |
| AdaptiveContextFusion | 4.5486 | 0.4274 | Multi-Scale Context Integration |

### Evaluation Benchmarks
The system evaluates architectures across diverse cognitive domains:
- **Reasoning**: ARC Challenge/Easy, HellaSwag, Physical IQA, Social IQA, Winogrande
- **Language Understanding**: BoolQ, OpenBookQA, SQuAD Completion
- **Memory & Context**: LambadaOpenAI sequence modeling
- **Specialized Tasks**: FDA domain-specific tasks, SWDE structured data extraction

## 🚀 Get Started

### System Requirements

- Python 3.8+
- MongoDB 4.4+
- Docker and Docker Compose
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM, 32GB recommended

### Package Installation

```bash
git clone https://github.com/your-org/ASI-Arch.git
cd ASI-Arch
conda create -n asi-arch python=3.10
conda activate asi-arch
pip install -r requirements.txt
```

### Core Dependencies Installation

```bash
# Install PyTorch with CUDA support
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# Install additional requirements
pip install -r database/requirements.txt
pip install -r rag/requirements.txt
```

### Environment Setup

1. **Configure Database Services**:
```bash
# Start MongoDB
cd database
chmod +x start_api.sh
./start_api.sh
```

2. **Launch RAG Service**:
```bash
# Configure OpenSearch and start RAG
cd rag
docker-compose up -d opensearch
sleep 30
python rag_api.py
```

3. **Configure Agent System**:
```bash
# Update configuration
cd agent
# Edit config.py with your service URLs
# RAG: "http://localhost:5000"
# DATABASE: "http://localhost:8000"
```

## 💻 Usage

### Running Architecture Evolution

```bash
# Execute single experiment cycle
cd agent
python pipeline.py
```

The pipeline automatically:
1. Samples a parent architecture using UCT algorithm from top-50 candidates
2. Evolves the architecture using AI agents with RAG-retrieved research insights
3. Validates and fixes code implementation issues
4. Trains and evaluates the new architecture across multiple benchmarks
5. Analyzes results and updates the candidate database

### Custom Architecture Evolution

```python
from agent.pipeline import run_single_experiment
import asyncio

# Run autonomous evolution cycle
async def evolve_architecture():
    success = await run_single_experiment()
    return success

# Execute evolution loop
result = asyncio.run(evolve_architecture())
```

### Continuous Evolution Mode

```bash
# Run continuous evolution (default mode)
cd agent
python pipeline.py  # Runs indefinitely, continuously evolving architectures
```

### RAG Service Usage

```bash
# Query research insights
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism improvements", "k": 5}'

# Get experiment statistics
curl http://localhost:5000/stats
```

### Database Operations

```python
from database.mongodb_database import create_client

# Access experimental data
client = create_client()
experiments = client.get_all_elements(limit=10)

# Get top-performing candidates
top_candidates = client.candidate_sample_from_range(1, 10, 5)

# Add new experimental results  
from database.element import DataElement
element = DataElement(
    time=datetime.now().isoformat(),
    name="experiment_name",
    result={"train": train_metrics, "test": test_metrics},
    program=source_code,
    motivation=design_motivation,
    analysis=performance_analysis
)
client.add_element_from_dict(element.to_dict())
```

## 📊 Evaluation

### Manual Evaluation

```python
# Evaluate specific architecture manually
from agent.eval.interface import evaluation
import asyncio

async def run_evaluation():
    success = await evaluation("architecture_name", "design_motivation")
    return success

result = asyncio.run(run_evaluation())
```

### Performance Analysis

```python
# Analyze experimental results manually
from agent.analyse.interface import analyse
import asyncio

async def run_analysis():
    result = await analyse("experiment_name", "motivation", parent=parent_index)
    return result

analysis = asyncio.run(run_analysis())
```

### UCT Candidate Sampling

```python
# Access UCT-based candidate sampling
from database.mongodb_database import create_client

client = create_client()
# Sample using UCT algorithm (rank 1-10, get 1 candidate)
parent_candidates = client.candidate_sample_from_range(1, 10, 1)
# Get reference candidates (rank 11-50, get 4 candidates)  
ref_candidates = client.candidate_sample_from_range(11, 50, 4)
```

## 🙏 Acknowledgement

ASI-Arch builds upon foundational work in neural architecture search, multi-agent systems, and automated machine learning. We particularly acknowledge the contributions of the research communities working on:

- Neural Architecture Search (NAS) and evolutionary algorithms
- Upper Confidence Trees (UCT) and Monte Carlo Tree Search methods
- Retrieval-Augmented Generation (RAG) systems for knowledge integration
- Multi-agent AI frameworks and agent orchestration
- Automated code generation and program synthesis
- Large language models for code understanding and generation

## ✍️ Citation

Please cite this repository if ASI-Arch contributes to your research:

```bibtex
@software{asi_arch_2025,
  title={ASI-Arch: Autonomous Neural Architecture Search through AI Agents},
  author={Your Research Team},
  year={2025},
  url={https://github.com/your-org/ASI-Arch},
  note={Framework for autonomous neural architecture search using UCT algorithm and AI agents}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Research Applications

ASI-Arch is designed for researchers and practitioners working on:

- Automated neural architecture search and discovery
- Evolutionary algorithms for architecture optimization
- UCT-based exploration strategies for design spaces
- AI-driven code generation and program synthesis
- Multi-agent system orchestration and collaboration
- RAG-enhanced knowledge integration for research
- Experimental AI research workflows and automation
- Large-scale architecture performance analysis

## 🛠️ Contributing

We welcome contributions to ASI-Arch! Please see our contributing guidelines and feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact the development team
- Check our documentation and examples 