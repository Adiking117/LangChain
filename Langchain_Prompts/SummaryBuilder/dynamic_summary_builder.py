import streamlit as st
from langchain_core.prompts import load_prompt
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model

st.set_page_config(
    page_title="📘 AI-Powered Research Explainer",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("<h1 style='text-align: center; color: #5A5A5A;'>📘 AI-Powered Research Explainer</h1>", unsafe_allow_html=True)
st.markdown("##### Generate concise, tailored explanations of foundational papers in AI.")

paper_input = st.selectbox(
    "📄 Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "🎨 Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "📏 Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template = load_prompt('template.json')

if st.button('🔍 Summarize'):
    with st.spinner("🧠 Thinking..."):
        chain = template | model
        result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
    st.success("✅ Here's your explanation:")
    st.markdown(f"<div style='background-color: #F8F9FA; padding: 15px; border-left: 4px solid #00BFFF;'>{result.content}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Crafted by Aditya with 🤖 | Powered by Tiny LLaMA")
