import streamlit as st
from openai import OpenAI
import base64

# 页面配置
st.set_page_config(page_title="我的AI助手", layout="wide")
st.title("🤖 我的专属AI助手")

# ---------------------- 侧边栏配置 ----------------------
with st.sidebar:
    st.header("⚙️ API设置")
    base_url = st.text_input(
        "API 接口地址",
        value="https://api.gemai.cc/v1"
    )
    model_name = st.text_input(
        "模型名称",
        value="[满血A]gemini-3-pro-preview-thinking-51"
    )
    st.divider()

    # 图片上传区域
    st.subheader("🖼️ 上传图片")
    uploaded_image = st.file_uploader(
        "选择图片文件",
        type=["png", "jpg", "jpeg", "webp"],
        help="支持PNG、JPG、WEBP格式，上传后输入问题即可提问"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="当前上传的图片", use_column_width=True)

    st.divider()

# 兼容本地&云端密钥
api_key = ""
try:
    api_key = st.secrets["API_KEY"]
    st.sidebar.success("✅ 密钥已加载，无需手动输入")
except Exception:
    api_key = st.sidebar.text_input("API 密钥", type="password", placeholder="请输入sk-开头的密钥")
    st.sidebar.caption("🔒 密钥仅在当前页面有效，不会被保存")


# ---------------------- 工具函数：图片转base64 ----------------------
def image_to_base64(image_file):
    """将上传的图片文件转为base64格式，用于API调用"""
    file_type = image_file.type
    base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{file_type};base64,{base64_str}"


# ---------------------- 聊天功能 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 如果消息包含图片，先显示图片
        if message.get("image_base64"):
            st.image(message["image_base64"], use_column_width=True)
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("输入你想问的问题..."):
    if not api_key:
        st.error("⚠️ 未检测到API密钥，请在左侧侧边栏填写！")
        st.stop()

    # 构造用户消息内容
    user_message = {"role": "user", "content": prompt}

    # 如果有上传图片，把图片加入消息
    image_base64 = None
    if uploaded_image:
        image_base64 = image_to_base64(uploaded_image)
        user_message["image_base64"] = image_base64

    # 保存到历史记录并显示
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        if image_base64:
            st.image(image_base64, use_column_width=True)
        st.markdown(prompt)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构造API请求的消息格式（兼容多模态）
        api_messages = []
        for msg in st.session_state.messages:
            if msg.get("image_base64"):
                # 带图片的消息，按OpenAI多模态格式组装
                api_messages.append({
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": msg["image_base64"]}}
                    ]
                })
            else:
                # 纯文本消息
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
            )
            response = st.write_stream(stream)

        # 保存AI回复到历史
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"调用失败：{str(e)}")
        st.info("请检查接口地址、模型名称和密钥是否正确；确保当前模型支持图片识别")