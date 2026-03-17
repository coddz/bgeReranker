import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HOME'] = r"F:\openclaw_models\extModels"

from FlagEmbedding import FlagReranker

print("Loading BGE Reranker model...")
model = FlagReranker(
    model_name_or_path="BAAI/bge-reranker-v2-m3",
    use_fp16=False
)
print("Model loaded successfully!")

# 测试用例
query = "什么是人工智能？"
passages = [
    "人工智能是计算机科学的一个分支，它试图创造出能够以类似人类智能的方式执行任务的系统。",
    "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
    "自然语言处理是人工智能的一个重要应用领域，旨在让计算机理解和处理人类语言。",
    "Python是一种广泛用于人工智能开发的编程语言。",
    "深度学习使用多层神经网络来处理复杂的数据模式，是人工智能的重要技术之一。"
]

print("\n测试重排序功能：")
print(f"查询：{query}")
print("\n原始段落：")
for i, p in enumerate(passages):
    print(f"{i+1}. {p}")

# 计算相关性分数
scores = model.compute_score([[query, p] for p in passages])
print("\n重排序结果（分数越高越相关）：")
# 排序结果
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
for idx, (orig_idx, score) in enumerate(ranked):
    print(f"{idx+1}. (得分：{score:.4f}) {passages[orig_idx]}")

print("\n✅ BGE Reranker 运行正常！")
