import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import tempfile

# ---------- 配置 ----------
# 从 Streamlit secrets 中读取敏感信息（部署时需要）
# 本地测试时，可以在 .streamlit/secrets.toml 中配置
try:
    API_KEY = st.secrets["DOUBAO_API_KEY"]
    MODEL_ID = st.secrets["MODEL_ID"]
except:
    # 本地测试备用（也可以直接填写，但注意不要上传到 GitHub）
    API_KEY = "你的豆包API Key"
    MODEL_ID = "你的接入点ID"

URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# 初始化 Chroma 客户端（持久化到磁盘）
client = chromadb.PersistentClient(path="./web_rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection_name = "doc_collection"

# 尝试获取或创建 collection
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
        st.error("不支持的文件类型，请上传 .txt 或 .pdf")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """将文本切分成多个片段，并尽量保持语义完整性"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 尽量在句号、感叹号、问号或换行处断开
            for i in range(min(end+20, len(text))-1, end-1, -1):
                if text[i] in '。！？\n':
                    end = i+1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def add_to_vector_store(chunks, file_name):
    """将文档片段存入向量库，每个片段附带元数据（来源文件名）"""
    # 为每个片段生成唯一ID
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    # 元数据：记录来源文件
    metadatas = [{"source": file_name} for _ in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)

def retrieve(query, top_k=3):
    """检索最相关的 top_k 个片段，并返回内容和来源"""
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    # 返回 (内容, 来源) 的列表
    return list(zip(documents, metadatas))

def ask_llm(query, context_chunks_with_source):
    """基于检索到的片段生成回答，同时返回使用的片段内容（用于显示来源）"""
    # 构建上下文文本（同时保存来源信息）
    context_parts = []
    sources = []
    for i, (doc, meta) in enumerate(context_chunks_with_source):
        context_parts.append(f"[片段{i+1}] {doc}")
        sources.append(f"片段{i+1} 来自：{meta.get('source', '未知')}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""请仅根据以下文档片段回答问题。如果文档中没有相关信息，请说“文档中未提及”。

文档片段：
{context}

问题：{query}

回答："""
    data = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return answer, sources
    else:
        return f"API错误: {response.status_code}", []

# ---------- Streamlit UI ----------
st.set_page_config(page_title="多文档智能问答", layout="wide")
st.title("📚 多文档智能问答 (RAG)")
st.markdown("上传多个 PDF/TXT 文档，然后提问，AI 将基于所有文档内容回答，并显示信息来源。")

# 侧边栏：多文件上传
with st.sidebar:
    st.header("📁 上传文档")
    uploaded_files = st.file_uploader("选择文件（支持多个）", type=["txt", "pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 处理文档"):
            total_chunks = 0
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"正在处理 {file.name} ..."):
                    text = load_text_from_file(file)
                    if text:
                        chunks = chunk_text(text)
                        if chunks:
                            cnt = add_to_vector_store(chunks, file.name)
                            total_chunks += cnt
                            st.success(f"{file.name} 已切分为 {len(chunks)} 个片段")
                progress_bar.progress((i+1)/len(uploaded_files))
            st.success(f"✅ 所有文档处理完成！共添加 {total_chunks} 个片段到知识库。")
            # 重置按钮状态
            st.session_state['docs_processed'] = True
    
    # 显示当前知识库中的文档数量
    if collection.count() > 0:
        st.info(f"当前知识库包含 {collection.count()} 个片段")

# 主区域：对话历史
st.header("💬 对话")
# 初始化会话状态（用于存储对话历史）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 信息来源"):
                for src in msg["sources"]:
                    st.caption(src)

# 输入框
if prompt := st.chat_input("请输入你的问题："):
    # 检查知识库是否为空
    if collection.count() == 0:
        st.warning("请先上传文档")
        st.stop()
    
    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 检索相关片段
    with st.spinner("检索中..."):
        retrieved = retrieve(prompt, top_k=3)
        if not retrieved:
            st.error("未找到相关文档片段")
            st.stop()
    
    # 生成回答
    with st.spinner("生成答案中..."):
        answer, sources = ask_llm(prompt, retrieved)
    
    # 显示 AI 回答
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("📖 信息来源"):
                for src in sources:
                    st.caption(src)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})