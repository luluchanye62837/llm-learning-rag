import os, dotenv
import numpy as np
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv('ZHIPUAI_API_KEY'),
                base_url='https://open.bigmodel.cn/api/paas/v4/')

# ===== 1. 准备语料:你的丝之歌"正确"资料 =====
docs = [
    "丝之歌中并没有'白沼'这个区域,正确的地名是'灰沼'。",
    "灰沼中的'劈鸟'不是Boss,劈鸟是一个跑图技巧:通过劈砍灰沼的飞鸟来绕过一段遭遇战。",
    "提高劈鸟成功率的关键,在于如何勾引两只鸟到合适的位置,而不是和它们硬战。",
    "最重要的是在进入这个房间后在背景有草的地方先等待，等到鸟（腐囊虫）冲刺时候同时向右冲刺起跳连续下批两只鸟即可"
]

# ===== 2. 写一个函数:把一段文字变成向量 =====
def get_embedding(text):
    resp = client.embeddings.create(
        model="embedding-3",
        input=text,            # 注意:智谱只能传单个字符串,不能传列表
        dimensions=1024        # 用1024维,够用且省token
    )
    return resp.data[0].embedding   # 取出那一串数字(一个长度1024的列表)

# ===== 3. 把所有语料逐段变成向量,存进一个列表 =====
print("正在把语料转成向量...")
doc_vectors = []
for d in docs:
    vec = get_embedding(d)
    doc_vectors.append(vec)
    print(f"  '{d[:15]}...' → 向量维度 {len(vec)}")

# 转成 numpy 数组,方便后面算相似度(Day1 的 numpy 上场)
doc_vectors = np.array(doc_vectors)
print(f"\n所有语料向量已就绪,形状: {doc_vectors.shape}")
print(f"第一段向量的前5个数字: {doc_vectors[0][:5]}")


# ===== 4. 余弦相似度函数(你的线代主场)=====
def cosine_similarity(a, b):
    # a·b 除以 (|a| * |b|)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # np.dot 算点积,np.linalg.norm 算向量的模(长度)

# ===== 5. 给定一个问题,找出最相关的 top-k 段 =====
def search(query, k=2):
    query_vec = np.array(get_embedding(query))

    # 批量算:一次性算出 query 和所有 doc 向量的余弦相似度,不用for循环
    # 分子:query 和每个 doc 的点积 → doc_vectors @ query_vec 得到一个数组
    # 分母:query的模 × 每个doc的模
    dots = doc_vectors @ query_vec                          # 矩阵×向量,一步算出所有点积
    doc_norms = np.linalg.norm(doc_vectors, axis=1)         # 每个doc向量的模(axis=1按行算)
    query_norm = np.linalg.norm(query_vec)
    sims = dots / (doc_norms * query_norm)                  # 一次性得到所有相似度

    # 取相似度最高的 k 个的下标
    top_idx = np.argsort(sims)[::-1][:k]                    # argsort升序排返回下标,[::-1]反转成降序

    results = []
    for i in top_idx:
        print(f"  [{sims[i]:.4f}] {docs[i]}")
        results.append(docs[i])
    return results

# # ===== 测试:问那个让模型翻车的问题 =====
# print("\n" + "="*40)
# question = "怎么提高白沼劈鸟的成功率？"
# print(f"问题: {question}\n")
# relevant = search(question)

# ===== 6. 把检索到的资料塞进 prompt,让模型看着资料回答 =====
def rag_answer(question):
    # 先检索(复用第5步的 search)
    relevant_docs = search(question, k=2)

    # 把检索到的资料拼成一段"参考资料"
    context = "\n".join(relevant_docs)

    # 关键:构造一个"开卷考试"的 prompt
    prompt = f"""请严格根据以下参考资料回答问题。如果资料中没有相关信息,就说"资料中没有提到",不要编造。

参考资料:
{context}

问题: {question}"""

    # 调对话模型(注意:这次不用 thinking,普通回答即可)
    response = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}}
    )
    return response.choices[0].message.content

# ===== 终极测试:同一个问题,这次用 RAG 回答 =====
print("\n" + "="*40)
print("【RAG 回答】")
answer = rag_answer("怎么提高白沼劈鸟的成功率？")
print(answer)