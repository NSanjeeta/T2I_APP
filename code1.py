import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="T2I App", layout="centered")

def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.jpeg")

col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.jpeg", width=80)

with col2:
    st.title("VarSa")
    st.write("Generate images using Stable Diffusion XL")

st.markdown("---")

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

st.markdown("""
<style>
.prompt-box {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="prompt-box">', unsafe_allow_html=True)

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Two confident young women standing together, professional photography",
    height=120
)

negative_prompt = st.text_input(
    "Negative prompt",
    value="blurry, distorted faces, extra limbs, low quality"
)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("🎨 Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        try:
            with st.spinner("Generating image..."):
                image = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                    negative_prompt=negative_prompt
                )

            st.image(image, caption="Generated Image", use_container_width=True)

        except Exception:
            st.error("Image generation failed. Please try again.")

st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Built with ❤️ using Streamlit & Stable Diffusion XL</p>",
    unsafe_allow_html=True
)
