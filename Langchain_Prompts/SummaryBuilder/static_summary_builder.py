import streamlit as st
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model

st.set_page_config(
    page_title="Research Summarizer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Research Summarizer</h1>", unsafe_allow_html=True)
st.markdown("#### Efficiently condense your text into key insights.")

user_input = st.text_area(
    label="✍️ Enter your research text:",
    placeholder="Paste or write your content here...",
    height=200,
)

if st.button('🔍 Summarize'):
    if user_input.strip():
        with st.spinner('Summarizing your text...'):
            result = model.invoke(user_input)
        st.success("✅ Summary:")
        st.markdown(f"<div style='background-color: #f0f0f5; padding: 10px; border-radius: 5px;'>{result.content}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to summarize.")

st.markdown("---")
st.caption("Crafted with ❤️ by Aditya | Powered by Tiny LLaMA")

