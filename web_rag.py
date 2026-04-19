import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import tempfile
import hashlib

# ---------- 配置 ----------
# 从 Streamlit secrets 读取（部署时）
try:
    API_KEY = st.secrets["DOUBAO_API_KEY"]
    MODEL_ID = st.secrets["MODEL_ID"]
except:
    # 本地测试时，可以硬编码（但不要提交到 GitHub）
    API_KEY = "你的API Key"
    MODEL_ID = "你的模型ID"

URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# 初始化 ChromaDB（持久化）
client = chromadb.PersistentClient(path="./web_rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection_name = "multi_docs"
try:
    collection = client.get_collection(name=collection_name, embedding_function=ef)
except:
    collection = client.create_collection(name=collection_name, embedding_function=ef)

# ---------- 辅助函数 ----------
def load_text_from_file(uploaded_file):
    """从上传的文件中提取文本（支持 txt 和 pdf）"""
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        os.unlink(tmp_path)
        return text
    else:
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """切分文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for i in range(min(end+20, len(text))-1, end-1, -1):
                if text[i] in '。！？\n':
                    end = i+1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def add_to_vector_store(chunks, doc_name):
    """将文档片段加入向量库，附带元数据（来源文档名）"""
    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        unique_id = hashlib.md5(f"{doc_name}_{i}".encode()).hexdigest()
        ids.append(unique_id)
        documents.append(chunk)
        metadatas.append({"source": doc_name, "chunk_index": i})
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return len(chunks)

def retrieve(query, top_k=3):
    """检索相关片段，同时返回元数据"""
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    return list(zip(documents, metadatas))

def ask_llm(query, context_chunks_with_meta):
    """基于检索到的片段生成回答，并返回来源信息"""
    context_parts = []
    sources = set()
    for i, (chunk, meta) in enumerate(context_chunks_with_meta):
        context_parts.append(f"[片段{i+1}] {chunk}")
        sources.add(meta.get("source", "未知文档"))
    context = "\n\n".join(context_parts)
    source_list = ", ".join(sources)
    
    prompt = f"""请仅根据以下文档片段回答问题。如果文档中没有相关信息，请说“文档中未提及”。

文档片段：
{context}

问题：{query}

回答："""
    data = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return answer, source_list
    else:
        return f"API错误: {response.status_code}", ""

# ---------- Streamlit UI ----------
st.set_page_config(page_title="多文档智能问答", layout="wide")
st.title("📚 多文档智能问答 (RAG)")
st.markdown("上传多个文档（TXT/PDF），然后提问，AI 将基于文档内容回答，并显示来源。")

# 侧边栏：文档上传与管理
with st.sidebar:
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader("上传文档（支持多个）", type=["txt", "pdf"], accept_multiple_files=True)
    if st.button("🚀 处理并构建知识库"):
        if uploaded_files:
            total_chunks = 0
            # 清空旧集合（简单处理：重建）
            client.delete_collection(collection_name)
            global collection
            collection = client.create_collection(name=collection_name, embedding_function=ef)
            
            for file in uploaded_files:
                with st.spinner(f"正在处理 {file.name} ..."):
                    text = load_text_from_file(file)
                    if text:
                        chunks = chunk_text(text)
                        cnt = add_to_vector_store(chunks, file.name)
                        total_chunks += cnt
                        st.success(f"{file.name} 已处理，添加 {cnt} 个片段")
                    else:
                        st.error(f"{file.name} 处理失败")
            st.success(f"知识库构建完成！共 {total_chunks} 个片段")
        else:
            st.warning("请先上传文档")
    
    if st.button("🗑️ 清空所有文档"):
        client.delete_collection(collection_name)
        collection = client.create_collection(name=collection_name, embedding_function=ef)
        st.success("已清空知识库")
        st.rerun()

# 主区域：对话
st.header("💬 对话")
# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📌 来源: {msg['sources']}")

# 用户输入
if prompt := st.chat_input("请输入问题"):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 检索
    with st.spinner("检索中..."):
        results = retrieve(prompt)
        if not results:
            answer = "知识库为空，请先上传文档。"
            sources = ""
        else:
            answer, sources = ask_llm(prompt, results)
    
    # 显示助手回复
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption(f"📌 来源: {sources}")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})