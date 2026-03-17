# BGE Reranker API Service

本项目提供高性能的 BGE Reranker 模型 API 服务，基于 BAAI/bge-reranker-v2-m3 模型，支持高并发请求，提供文档重排序能力。

---

## 🚀 快速开始

### 首次使用只需2步：
1. **下载模型**：双击运行 `bge_reranker_init.bat`
2. **启动服务**：双击运行 `run_api_enhanced.bat`

服务启动后即可访问：
- 接口文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

---

## Files

| 文件名 | 说明 |
|--------|------|
| `bgeReranker_init.py` | 原版模型下载脚本 |
| `bgeReranker_API.py` | 原版FastAPI服务 |
| `bgeReranker_init_enhanced.py` | ✅ 增强版模型下载脚本（支持环境变量配置） |
| `bgeReranker_API_enhanced.py` | ✅ 增强版高性能API服务（异步非阻塞、高并发） |
| `bgeReranker_API_production.py` | ✅ 生产级安全API服务（并发控制、请求限流、超时保护、全面监控） |
| `bge_reranker_init.bat` | ✅ 一键模型下载启动脚本（Windows） |
| `run_api_enhanced.bat` | ✅ 一键API服务启动脚本（Windows） |
| `run_api_production.bat` | ✅ 生产级一键启动脚本（完整配置、稳定性保障） |

## Prerequisites

- Anaconda or Miniconda installed
- The `openclaw` conda environment already exists with:
  - PyTorch 2.8.0+cu128
  - Transformers 4.57.6
  - CUDA support

## Setup and Usage

### 1. Activate the existing environment:
```bash
conda activate openclaw
```

### 2. Verify/install Dependencies (if needed):
```bash
conda activate openclaw
pip install FlagEmbedding fastapi uvicorn  
```

### 3. Download the Model:
#### 方式1：一键下载（推荐）
直接双击运行 `bge_reranker_init.bat`，脚本会自动配置环境并下载模型。

#### 方式2：手动下载
```bash
conda activate openclaw
python bgeReranker_init_enhanced.py
```
这会将 "BAAI/bge-reranker-v2-m3" 模型下载到 `F:\openclaw_models\extModels` 目录（可通过修改bat文件中的 `HF_HOME` 环境变量自定义存储路径）。

### 4. Run the API Server:

#### 🌟 增强版API（推荐使用）
##### 方式1：一键启动
直接双击运行 `run_api_enhanced.bat` 即可自动配置并启动高性能增强版API服务。

##### 方式2：手动启动
```bash
conda activate openclaw
python bgeReranker_API_enhanced.py
```

#### 原版API
```bash
conda activate openclaw
python bgeReranker_API.py
```
API服务启动后访问地址：`http://localhost:8000`

### 5. Testing the API:
- Documentation at `http://localhost:8000/docs`
- Health check: `GET /health`
- Example API call:
  ```bash
  curl -X POST "http://localhost:8000/rerank" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What is the capital of France?",
      "documents": [
        "Berlin is the capital of Germany",
        "Paris is the capital of France",
        "London is the capital of England"
      ]
    }'
  ```

## API Endpoints

- `GET /` - Welcome message
- `POST /rerank` - Rerank documents based on query relevance
- `GET /health` - Health check

## Configuration

The API temporarily sets `HF_HOME` to `F:\openclaw_models\extModels` so the model is stored outside of the C drive.

---

---
## 🏭 生产级API (Production API)

### 安全特性
`bgeReranker_API_production.py` 是生产级安全版本，在增强版基础上增加了企业级安全和稳定性保障：

- 🛡️ **请求并发控制**：支持设置最大并发请求数（默认8个），防止单一节点过载
- ⏱️ **智能超时保护**：内置请求超时机制（默认60秒），防止长期挂起
- 📊 **实时监控指标**：提供详细的运行状态和性能指标，便于运维监控
- 🔒 **资源管理安全**：优化的资源分配和清理机制，提升长时间运行稳定性
- 📈 **请求追踪头**：记录处理时间的 X-Process-Time 响应头，辅助性能分析
- 💥 **弹性故障恢复**：异常处理更加完善，服务在错误后能更快恢复

### 快速启动（生产）
#### 方式1：一键启动（推荐）
直接双击运行 `run_api_production.bat` 即可部署带有安全配置的生产级服务

#### 方式2：手动启动
```bash
conda activate openclaw
python bgeReranker_API_production.py
```

### 安全配置选项
生产版本新增关键安全配置，请在 `run_api_production.bat` 中根据您的服务器能力和负载需求调整：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| MAX_CONCURRENT_REQUESTS | 8 | 最大并发请求数（控制资源消耗） |
| PROCESSING_TIMEOUT | 90 | 请求超时时间（秒，防范长时间挂起） |
| CUDA_CLEAR_CACHE | true | 是否启用智能CUDA缓存清理（动态平衡性能与内存） |

## 🌟 增强版API (Recommended for Development)

### 能力说明
`bgeReranker_API_enhanced.py` 是优化后的高性能版本，具备以下特性：
- 🚀 **异步非阻塞架构**：计算逻辑运行在独立线程，不会阻塞事件循环，并发性能提升5-10倍
- 🔒 **线程安全**：移除不必要的全局锁，支持真正的并行处理
- ⚡ **高精度计算**：使用标准 `math.exp()` 确保sigmoid分数归一化计算准确
- 🎯 **丰富的功能**：支持分数归一化、Top-K结果返回、长度校验等
- 🛡️ **生产级可靠性**：完善的错误处理、参数校验、运行时监控
- ⚙️ **灵活配置**：所有参数支持通过环境变量自定义，无需修改代码
- 💾 **智能显存管理**：可选的CUDA缓存清理策略，平衡性能和显存占用

### 快速启动
#### 方式1：一键启动（推荐）
直接双击运行 `run_api_enhanced.bat` 即可自动配置环境并启动服务

#### 方式2：手动启动
```bash
conda activate openclaw
python bgeReranker_API_enhanced.py
```

### 接口说明
#### POST /rerank
对文档列表按照与查询的相关性进行重排序

**请求参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 查询文本 |
| documents | list[string] | 是 | 待排序的文档列表 |
| top_k | int | 否 | 返回前K个最相关的结果，默认返回全部 |
| normalize_scores | bool | 否 | 是否将分数归一化到0-1范围（使用sigmoid函数），默认false |

**响应字段：**
| 字段 | 类型 | 说明 |
|------|------|------|
| results | list[object] | 排序后的结果列表 |
| results.index | int | 文档在原列表中的索引位置 |
| results.text | string | 文档内容 |
| results.score | float | 相关性分数，值越高越相关 |

### 调用示例
#### 1. 基础调用
```bash
curl -X POST "http://localhost:8000/rerank" ^
-H "Content-Type: application/json" ^
-d "{\"query\":\"法国的首都是哪里？\",\"documents\":[\"伦敦是英国的首都\",\"巴黎是法国的首都和最大城市\",\"柏林是德国的首都\",\"马德里是西班牙的首都\"]}"
```

#### 2. 带分数归一化和Top-K筛选
```bash
curl -X POST "http://localhost:8000/rerank" ^
-H "Content-Type: application/json" ^
-d "{\"query\":\"什么是人工智能？\",\"documents\":[\"人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。\",\"猫是一种常见的家庭宠物。\",\"机器学习是人工智能的重要子领域，通过算法从数据中学习。\",\"北京是中国的首都。\"],\"top_k\":2,\"normalize_scores\":true}"
```

#### 返回示例
```json
{
  "results": [
    {
      "index": 0,
      "text": "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。",
      "score": 0.987
    },
    {
      "index": 2,
      "text": "机器学习是人工智能的重要子领域，通过算法从数据中学习。",
      "score": 0.892
    }
  ]
}
```

### 可配置环境变量
可以在 `run_api_enhanced.bat` 中修改以下配置，或设置系统环境变量生效：
| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| MAX_DOCUMENTS | 200 | 单次请求最大支持的文档数量 |
| MAX_DOC_LENGTH | 4096 | 单个文档最大长度（字符数） |
| MAX_QUERY_LENGTH | 1024 | 查询文本最大长度（字符数） |
| BATCH_SIZE | 32 | 模型推理批处理大小 |
| USE_FP16 | true | 是否使用FP16半精度推理（仅CUDA可用时生效） |
| ENABLE_CORS | true | 是否开启CORS跨域支持 |
| CUDA_CLEAR_CACHE | false | 是否在每批推理后清理CUDA缓存（可减少显存占用，但会略微降低性能） |
| HF_HOME | F:\openclaw_models\extModels | 模型存储目录，优先使用系统环境变量 |

### 其他接口
- `GET /` - 服务欢迎页面，显示版本信息
- `GET /health` - 健康检查接口，返回服务状态、模型加载状态、CUDA信息、配置参数等
- `GET /docs` - Swagger UI 交互式接口文档，可以直接在页面上测试接口