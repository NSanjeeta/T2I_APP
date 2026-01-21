import streamlit as st
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="T2I App", layout="centered")

col1, col2 = st.columns([1, 6])

with col1:
    if os.path.exists("logo.jpeg"):
        st.image("logo.jpeg", width=80)

with col2:
    st.title("VarSa T2I-GPT")
    st.write("Generate images using Stable Diffusion XL")

HF_TOKEN = st.secrets.get("HF_TOKEN")

if HF_TOKEN is None:
    st.error("HF_TOKEN not found. Please add it in Streamlit Secrets.")
    st.stop()

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Two confident young women standing together, professional photography",
    height=140
)

if st.button("🎨 Generate Image"):
    if prompt.strip() == "":
        st.warning("Enter a prompt.")
    else:
        try:
            with st.spinner("Generating image..."):
                image = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0"
                )

            st.image(image, caption="Generated Image", use_container_width=True)

        except Exception:
            st.error("Image generation failed. Please try again.")
