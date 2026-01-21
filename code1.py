import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="T2I App")

st.title("VSK")
st.write("Generate images using Stable Diffusion XL")

# Hugging Face token from Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Two confident young women standing together, professional photography"
)

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image = client.text_to_image(
                prompt,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                negative_prompt="blurry, distorted faces, extra limbs, low quality"
            )

        st.image(image, caption="Generated Image", use_container_width=True)
