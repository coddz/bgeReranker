# 快速启动指南

## 1. 环境准备

在开始之前，请确保您的环境中已安装：
- Anaconda 或 Miniconda
- 已创建名为 `openclaw` 的 conda 环境
- 该环境包含 PyTorch 2.8.0+cu128 和 Transformers 4.57.6

## 2. 模型下载

首次使用需下载 BGE 重排序模型，在 conda `openclaw` 环境中运行：

```bash
# 方法1：使用便捷脚本（推荐）
./bge_reranker_init.bat

# 方法2：直接运行初始化脚本
conda activate openclaw
python bgeReranker_init_enhanced.py
```

## 3. API 服务启动（按需选择）

### 生产模式 - 生产级安全版（推荐用于生产环境）：
```bash
# Windows 一键启动
./run_api_production.bat

# Linux/macOS 启动命令
conda activate openclaw
python bgeReranker_API_production.py
```

### 开发模式 - 增强版（推荐用于开发/中等负载）：
```bash
# Windows 一键启动
./run_api_enhanced.bat

# Linux/macOS 启动命令
conda activate openclaw
python bgeReranker_API_enhanced.py
```

## 4. API 接口访问

根据启动的版本，服务将运行在 `http://localhost:8000`

可用端点：
- `GET /` - 服务欢迎页面，显示版本及配置信息
- `POST /rerank` - 核心重排序接口
- `GET /health` - 健康检查，返回服务状态
- `GET /docs` - Swagger UI 接口文档
- `GET /metrics` - （仅生产版）服务监控指标

## 5. 重排序 API 调用示例

```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "documents": [
      "AI is a branch of computer science...",
      "Cats are common household pets.",
      "Machine learning is a subset of AI...",
      "Beijing is the capital of China."
    ],
    "top_k": 2,
    "normalize_scores": true
  }'
```

## 6. 版本区别

- `bgeReranker_API.py` - 基础API版本
- `bgeReranker_API_enhanced.py` - 异步高并发版本（开发环境推荐）
- `bgeReranker_API_production.py` - 生产安全版本（含并发控制、超时、完整监控）         |

这些环境变量可以方便地在 `run_api_enhanced.bat` 中配置。