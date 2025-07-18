# AI_Archer

AI_Archer 是一个基于人工智能的自动化代码进化和优化系统，通过智能化的流水线实现代码的自动采样、进化、评估和分析。

## 项目概述

AI_Archer 采用模块化设计，通过多个智能代理协同工作，实现代码的自动化进化和优化。系统能够从数据库中采样程序，通过AI模型进行代码进化，评估生成的代码质量，并进行深度分析。

## 核心功能

### 🔄 自动化流水线
- **程序采样**: 从数据库中智能采样程序作为进化基础
- **代码进化**: 基于上下文和模式进行代码创建和优化
- **程序评估**: 自动评估生成代码的质量和性能
- **结果分析**: 深度分析实验结果并生成报告

### 🧠 智能代理系统
- **Creator**: 负责创建新的代码实现
- **Optimizer**: 优化现有代码性能和结构
- **Code Checker**: 验证代码正确性和质量
- **Motivation Checker**: 检查进化动机的合理性

## 项目结构

```
AI_Archer/
├── agent/                    # 核心代理模块
│   ├── analyse/             # 分析模块
│   │   ├── interface.py     # 分析接口
│   │   ├── model/          # 分析模型
│   │   └── prompts/        # 分析提示词
│   ├── database/           # 数据库模块
│   ├── eval/               # 评估模块
│   │   ├── interface.py    # 评估接口
│   │   ├── model/         # 评估模型
│   │   └── prompts/       # 评估提示词
│   ├── evolve/            # 进化模块
│   │   ├── interface.py   # 进化接口
│   │   ├── model/        # 进化模型
│   │   └── prompt/       # 进化提示词
│   ├── files/            # 文件存储
│   ├── tools/            # 工具模块
│   ├── utils/            # 工具函数
│   ├── config.py         # 配置文件
│   └── pipeline.py       # 主流水线
└── requirements.txt      # 依赖包列表
```

## 安装和配置

### 1. 环境要求
- Python 3.8+
- 支持异步操作的环境

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置设置
编辑 `agent/config.py` 文件，配置以下参数：

```python
class Config:
    # 目标文件
    SOURCE_FILE: str = "evolve file"
    
    # 训练脚本
    BASH_SCRIPT: str = "your training script"
    
    # 实验结果文件
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"
    
    # 调试文件
    DEBUG_FILE: str = "./files/debug/training_error.txt"
    
    # 代码池目录
    CODE_POOL: str = "./pool"
    
    # 最大调试尝试次数
    MAX_DEBUG_ATTEMPT: int = 3
    
    # 最大重试次数
    MAX_RETRY_ATTEMPTS: int = 10
    
    # RAG服务URL
    RAG: str = "your rag url"
    
    # 数据库URL
    DATABASE: str = "your database url"
```

## 使用方法

### 运行单次实验
```python
import asyncio
from agent.pipeline import run_single_experiment

# 运行单次实验
async def main():
    success = await run_single_experiment()
    if success:
        print("实验成功完成")
    else:
        print("实验失败")

asyncio.run(main())
```

### 运行完整流水线
```python
import asyncio
from agent.pipeline import main

# 运行完整流水线
asyncio.run(main())
```

## 核心模块说明

### 进化模块 (evolve)
- **创建模式**: 基于上下文创建全新的代码实现
- **优化模式**: 对现有代码进行性能和结构优化
- **代码检查**: 验证生成代码的正确性
- **动机检查**: 确保进化动机的合理性和唯一性

### 评估模块 (eval)
- 自动评估生成代码的质量
- 运行测试用例验证功能正确性
- 性能基准测试
- 生成评估报告

### 分析模块 (analyse)
- 深度分析实验结果
- 生成详细的分析报告
- 识别性能瓶颈和优化建议
- 跟踪进化历史和趋势

### 数据库模块 (database)
- 程序采样和存储
- 实验结果持久化
- 进化历史记录
- 性能指标跟踪

## 日志系统

系统提供完整的日志记录功能：
- 流水线级别日志
- 步骤级别日志
- 错误和警告日志
- 代理运行日志

## 扩展和自定义

### 添加新的进化策略
1. 在 `evolve/model/` 中添加新的模型
2. 在 `evolve/prompt/` 中添加相应的提示词
3. 在 `evolve/interface.py` 中注册新策略

### 添加新的评估指标
1. 在 `eval/model/` 中实现新的评估模型
2. 在 `eval/prompts/` 中添加评估提示词
3. 更新 `eval/interface.py` 中的评估逻辑


## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 Issue
- 提交 Pull Request
- 发送邮件至项目维护者

