import streamlit as st
from openai import OpenAI
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

# 页面配置
st.set_page_config(page_title="象棋AI大师", layout="wide")
st.title("♟️ 中国象棋专属AI大师")

# ---------------------- 侧边栏配置 ----------------------
with st.sidebar:
    # 1. API基础设置
    st.header("⚙️ API设置")
    base_url = st.text_input("API 接口地址", value="https://api.gemai.cc/v1")
    model_name = st.text_input("聊天模型名称", value="[满血A]gemini-3-pro-preview-thinking-51")
    embedding_model = st.text_input("嵌入模型名称", value="text-embedding-3-small",
                                    help="用于知识库检索，找API站里的嵌入模型名")
    st.divider()

    # 2. RAG知识库设置
    st.header("📚 象棋知识库(RAG)")
    enable_rag = st.checkbox("启用象棋知识库", value=True)
    uploaded_files = st.file_uploader(
        "上传象棋文档（支持PDF/TXT）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="上传象棋棋谱、残局教程、开局理论等文档，AI会优先参考文档内容回答"
    )

    if uploaded_files and st.button("🔨 构建知识库", type="primary"):
        with st.spinner("正在处理文档、构建象棋知识库..."):
            # 读取并合并所有文档
            all_docs = []
            for file in uploaded_files:
                # 临时保存文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                # 加载文档
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                os.unlink(tmp_path)

            # 文本切片
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", " ", ""]
            )
            split_docs = text_splitter.split_documents(all_docs)

            # 构建向量库
            api_key = st.secrets.get("API_KEY", "")
            if not api_key:
                api_key = st.session_state.get("input_api_key", "")

            if not api_key:
                st.error("请先配置API密钥再构建知识库")
                st.stop()

            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url=base_url,
                model=embedding_model
            )
            vector_store = FAISS.from_documents(split_docs, embeddings)
            st.session_state["vector_store"] = vector_store
            st.success(f"✅ 知识库构建完成！共处理 {len(split_docs)} 条知识片段")

    st.divider()

    # 3. 图片上传（保留原多模态功能）
    st.subheader("🖼️ 上传棋谱图片")
    uploaded_image = st.file_uploader(
        "选择棋谱图片",
        type=["png", "jpg", "jpeg", "webp"],
        help="上传棋盘图片，AI可以识别局面、给出走法建议"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="当前上传的棋谱", use_column_width=True)

# ---------------------- 密钥兼容处理 ----------------------
api_key = ""
try:
    api_key = st.secrets["API_KEY"]
except Exception:
    api_key = st.sidebar.text_input("API 密钥", type="password", key="input_api_key")
    st.sidebar.caption("🔒 密钥仅在当前页面有效，不会被保存")


# ---------------------- 工具函数 ----------------------
def image_to_base64(image_file):
    """图片转base64，用于多模态调用"""
    file_type = image_file.type
    base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{file_type};base64,{base64_str}"


def retrieve_knowledge(query, top_k=3):
    """从知识库检索相关内容"""
    if "vector_store" not in st.session_state:
        return ""
    docs = st.session_state["vector_store"].similarity_search(query, k=top_k)
    context = "\n---\n".join([doc.page_content for doc in docs])
    return context


# ---------------------- 聊天核心逻辑 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_base64"):
            st.image(message["image_base64"], use_column_width=True)
        st.markdown(message["content"])

# 处理用户提问
if prompt := st.chat_input("问我任何象棋问题，比如：屏风马怎么应对中炮？"):
    if not api_key:
        st.error("⚠️ 请先在左侧侧边栏填写API密钥！")
        st.stop()

    # 构造用户消息
    user_message = {"role": "user", "content": prompt}
    image_base64 = None
    if uploaded_image:
        image_base64 = image_to_base64(uploaded_image)
        user_message["image_base64"] = image_base64
    st.session_state.messages.append(user_message)

    # 显示用户消息
    with st.chat_message("user"):
        if image_base64:
            st.image(image_base64, use_column_width=True)
        st.markdown(prompt)

    # RAG检索知识
    context = ""
    if enable_rag and "vector_store" in st.session_state:
        context = retrieve_knowledge(prompt)
        system_prompt = f"""你是专业的中国象棋特级大师，精通象棋规则、开局体系、中局战术、残局杀法、古今棋谱。
请优先参考下面的知识库内容回答用户的问题，结合专业象棋术语，给出准确、实用的解答。
如果知识库内容不足，可以结合你的专业知识补充，但不要编造棋谱和规则。

【知识库参考内容】
{context}
"""
    else:
        system_prompt = "你是专业的中国象棋特级大师，精通象棋各类知识，用专业易懂的方式回答用户问题。"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构造API请求消息
        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.messages:
            if msg.get("image_base64"):
                api_messages.append({
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": msg["image_base64"]}}
                    ]
                })
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # 流式输出回答
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
                temperature=0.3
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"调用失败：{str(e)}")
        st.info("请检查密钥、模型名称是否正确；嵌入模型需选择API站支持的版本")