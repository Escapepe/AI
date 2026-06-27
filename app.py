import streamlit as st
from openai import OpenAI

# 页面配置
st.set_page_config(page_title="我的AI助手", layout="wide")
st.title("🤖 我的专属AI助手")

# ---------------------- 侧边栏配置 ----------------------
with st.sidebar:
    st.header("⚙️ API设置")
    # 默认填好哈基米API的地址，可手动修改
    base_url = st.text_input(
        "API 接口地址",
        value="https://api.gemai.cc/v1"
    )
    # 默认填好常用模型，可手动修改
    model_name = st.text_input(
        "模型名称",
        value="gemini-2.5-pro"
    )
    st.divider()
    st.caption("✅ 默认适配哈基米API站")
    st.caption("🔒 密钥已在后台配置，无需手动输入")

# 读取后台密钥（优先用后台配置的，没有再提示输入）
api_key = st.secrets.get("API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("API 密钥", type="password")

# ---------------------- 聊天功能 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("输入你想问的问题..."):
    if not api_key:
        st.error("⚠️ 未检测到API密钥，请在侧边栏输入或在后台配置")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"调用失败：{str(e)}")
        st.info("请检查接口地址、模型名称和密钥是否正确")