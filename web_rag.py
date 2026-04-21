import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import tempfile
import hashlib
from docx import Document
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ========== 加载环境变量 ==========
load_dotenv()  # 自动寻找当前目录下的 .env 文件

# 优先从环境变量读取，其次从 st.secrets（云端）
API_KEY = os.getenv("DOUBAO_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

# 如果本地没有，尝试从 streamlit secrets 读取（部署时）
if not API_KEY or not MODEL_ID:
    try:
        API_KEY = st.secrets["DOUBAO_API_KEY"]
        MODEL_ID = st.secrets["MODEL_ID"]
    except:
        pass

# 如果都没有，报错
if not API_KEY or not MODEL_ID:
    st.error("请在 .env 文件或 Streamlit secrets 中设置 DOUBAO_API_KEY 和 MODEL_ID")
    st.stop()

# ========== 配置 API ==========
URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# ========== 初始化 ChromaDB ==========
client = chromadb.PersistentClient(path="./web_rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection_name = "multi_docs"
try:
    collection = client.get_collection(name=collection_name, embedding_function=ef)
except:
    collection = client.create_collection(name=collection_name, embedding_function=ef)

# ========== 辅助函数 ==========
def load_text_from_file(uploaded_file):
    """支持 txt, pdf, docx, md"""
    name = uploaded_file.name
    if name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        text = "\n".join([page.extract_text() for page in reader.pages])
        os.unlink(tmp_path)
        return text
    elif name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith('.md'):
        md_text = uploaded_file.read().decode('utf-8')
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    else:
        return None

def chunk_text(text, chunk_size=300, overlap=50):
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
    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{doc_name}_{i}".encode()).hexdigest()
        ids.append(uid)
        documents.append(chunk)
        metadatas.append({"source": doc_name, "index": i})
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return len(chunks)

def retrieve(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    return list(zip(docs, metas))

def ask_llm(query, context_chunks_with_meta):
    context_parts = []
    sources = set()
    for i, (chunk, meta) in enumerate(context_chunks_with_meta):
        context_parts.append(f"[{i+1}] {chunk}")
        sources.add(meta.get("source", "未知"))
    context = "\n\n".join(context_parts)
    source_str = ", ".join(sources)
    prompt = f"""请仅根据以下文档片段回答问题。如果文档中没有相关信息，请说“文档中未提及”。

文档片段：
{context}

问题：{query}

回答："""
    data = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(URL, headers=HEADERS, json=data, timeout=30)
        if resp.status_code == 200:
            ans = resp.json()["choices"][0]["message"]["content"]
            return ans, source_str
        else:
            return f"API错误: {resp.status_code}", ""
    except Exception as e:
        return f"请求异常: {str(e)}", ""

# ========== Streamlit UI ==========
st.set_page_config(page_title="多文档智能问答", layout="wide")
st.title("📚 多文档智能问答 (RAG)")
st.markdown("上传多个文档（TXT/PDF/DOCX/MD），然后提问。")

with st.sidebar:
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader("选择文件（多选）", type=["txt","pdf","docx","md"], accept_multiple_files=True)
    if st.button("🚀 处理并构建知识库"):
        if uploaded_files:
            total = 0
            # 清空旧数据
            client.delete_collection(collection_name)
            collection = client.create_collection(name=collection_name, embedding_function=ef)
            for file in uploaded_files:
                with st.spinner(f"处理 {file.name} ..."):
                    text = load_text_from_file(file)
                    if text:
                        chunks = chunk_text(text)
                        cnt = add_to_vector_store(chunks, file.name)
                        total += cnt
                        st.success(f"{file.name} → {cnt} 个片段")
                    else:
                        st.error(f"{file.name} 处理失败")
            st.success(f"知识库构建完成，共 {total} 个片段")
        else:
            st.warning("请先上传文件")
    if st.button("🗑️ 清空所有文档"):
        client.delete_collection(collection_name)
        collection = client.create_collection(name=collection_name, embedding_function=ef)
        st.success("已清空知识库")
        st.rerun()
    if st.button("🧹 清除对话历史"):
        st.session_state.messages = []
        st.success("对话历史已清除")
        st.rerun()

st.header("💬 对话")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📌 来源: {msg['sources']}")
        if "retrieved" in msg and msg["retrieved"]:
            with st.expander("📖 参考片段"):
                for i, r in enumerate(msg["retrieved"]):
                    st.text(f"{i+1}. {r[:150]}...")

if prompt := st.chat_input("请输入问题"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("检索并思考中..."):
        results = retrieve(prompt)
        if not results:
            answer = "知识库为空，请先上传文档。"
            sources = ""
            retrieved_chunks = []
        else:
            answer, sources = ask_llm(prompt, results)
            retrieved_chunks = [chunk for chunk, _ in results]

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption(f"📌 来源: {sources}")
        if retrieved_chunks:
            with st.expander("📖 参考的文档片段"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.text(f"[片段{i+1}] {chunk[:200]}...")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieved": retrieved_chunks
    })