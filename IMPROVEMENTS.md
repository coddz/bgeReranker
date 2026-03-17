# BGE Reranker API 改进说明

## 主要改进内容

### 1. 线程安全改进
- 添加了 `model_lock = asyncio.Lock()` 全局锁来保护模型访问
- 在 `rerank` 函数和模型操作中使用异步锁，防止并发导致的竞态条件

### 2. 输入验证增强
- 添加了每篇文档最大长度限制 (`MAX_DOC_LENGTH`)
- 添加了查询字符串最大长度限制 (`MAX_QUERY_LENGTH`)
- 在处理前验证每片文档的长度
- 验证查询字符串的长度

### 3. 内存管理优化
- 实现了 `batch_compute_score()` 函数进行批处理
- 防止由于大量文档一次性处理造成的内存不足
- 在批处理间隙执行GPU内存清理

### 4. 健壮性改进
- 在启动时对模型进行预热测试
- 为健康检查端点增加了更多配置参数的展示
- 添加了更详细的错误处理和日志记录

### 5. 代码质量改进
- 修正了可能导致运行时错误的设备属性访问 `model_device.type`
- 使用类型安全的设备属性访问方式
- 增强了代码的鲁棒性

### 6. 性能优化
- 添加了批量大小配置 (`BATCH_SIZE`) 来控制内存使用
- 改进了GPU内存管理机制
- 通过环境变量可调节的配置参数

## 环境变量配置

所有新添加的配置都可以通过环境变量设置，建议使用 `run_api_enhanced.bat` 启动：

- `MAX_DOC_LENGTH`: 单个文档最大长度 (默认: 4096字符)
- `MAX_QUERY_LENGTH`: 查询最大长度 (默认: 1024字符)
- `BATCH_SIZE`: 批处理大小 (默认: 32)
- `MAX_DOCUMENTS`: 每次请求最大文档数 (原功能保留: 默认100)
- `HOST`: 绑定IP地址 (原功能保留: 默认0.0.0.0)
- `PORT`: 绑定端口 (原功能保留: 默认8000)
- `USE_FP16`: 是否使用半精度 (原功能保留: 默认自动)
- `ENABLE_CORS`: 是否启用跨域 (原功能保留: 默认true)

## 启动方式

使用 `run_api_enhanced.bat` 文件启动改进版API：
```
双击运行 run_api_enhanced.bat
```

或者手动设置环境变量后运行:
```
set MAX_DOCUMENTS=200
set MAX_DOC_LENGTH=4096
set MAX_QUERY_LENGTH=1024
set BATCH_SIZE=32
set USE_FP16=true
python bgeReranker_API_enhanced.py
```

## 向后兼容性

本增强版 API 100% 兼容原版 API 的所有功能和接口，但增加了一些额外的健壮性措施。

## API 接口

所有 API 接口保持不变：
- `GET /` - 欢迎信息
- `POST /rerank` - 核心重排功能
- `GET /health` - 健康检查

## 错误代码

错误返回码与原版保持一致，仅在验证逻辑中增加了更详细的错误信息描述。