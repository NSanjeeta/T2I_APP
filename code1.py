import streamlit as st
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="T2I App", layout="wide")

def set_background(image_path):
    if not os.path.exists(image_path):
        return
    st.markdown(
        f"""
        <style>
        html, body {{
            height: 100%;
        }}
        .stApp {{
            background-image: url("{image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.jpeg")

col1, col2 = st.columns([1, 6])

with col1:
    if os.path.exists("logo.jpeg"):
        st.image("logo.jpeg", width=80)

with col2:
    st.title("VarSa T2I-GPT")
    st.write("Generate images using Stable Diffusion XL")

st.markdown("---")

HF_TOKEN = st.secrets.get("HF_TOKEN")

if HF_TOKEN is None:
    st.error("Hugging Face token not found. Please add HF_TOKEN in Streamlit Secrets.")
    st.stop()

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

st.markdown("""
<style>
.prompt-box {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 24px;
    border-radius: 14px;
    max-width: 900px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="prompt-box">', unsafe_allow_html=True)

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Two confident young women standing together, professional photography",
    height=140
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎨 Generate Image"):
    if prompt.strip() == "":
        st.warning("Enter a prompt.")
    else:
        try:
            with st.spinner("Generating image..."):
                image = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                    negative_prompt="blurry, distorted faces, extra limbs, low quality"
                )

            st.image(image, caption="Generated Image", use_container_width=True)

        except Exception:
            st.error("Image generation failed. Please try again.")
