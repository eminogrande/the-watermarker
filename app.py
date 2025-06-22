import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList
)
# The watermark processor and detector have been moved to a separate library
# We will install it and import from there.
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_model_and_tokenizer(model_name="gpt2"):
    """Loads and caches a model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# --- Core Application Logic ---
st.set_page_config(layout="wide", page_title="Real LLM Watermarker")
st.title("Real Probabilistic Watermarking Tool")
st.info("This application uses a real LLM (`gpt2`) running locally to demonstrate verifiable watermarking.")

# --- Load the Model --- 
with st.spinner("Loading local LLM (`gpt2`). This may take a moment on first run..."):
    model, tokenizer = load_model_and_tokenizer()
st.success("Model `gpt2` is loaded and ready!")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Paraphrase & Embed Watermark")
    secret_key = st.text_input("Enter Your Secret Key", value="correct horse battery staple")
    text_to_paraphrase = st.text_area(
        "Enter text to paraphrase:",
        "Modern generative AI can produce high-quality text, but it is difficult to determine the provenance of such content.",
        height=150
    )

    if st.button("Run Paraphrase and Watermark"):
        if not text_to_paraphrase or not secret_key:
            st.warning("Please provide text and a secret key.")
        else:
            with st.spinner("Generating text and embedding your watermark..."):
                # This is the correct initialization
                watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                             gamma=0.25,
                                                             delta=2.0,
                                                             seeding_scheme="simple_1",
                                                             hash_key=hash(secret_key))

                inputs = tokenizer(text_to_paraphrase, return_tensors="pt", add_special_tokens=False)
                
                output = model.generate(
                    **inputs,
                    logits_processor=LogitsProcessorList([watermark_processor]),
                    max_new_tokens=200,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                watermarked_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                st.session_state['generated_text'] = watermarked_text

    if 'generated_text' in st.session_state:
        st.subheader("Your Paraphrased and Watermarked Text:")
        st.text_area("Output Text", st.session_state.generated_text, height=200)

with col2:
    st.header("2. Detect Watermark")
    key_to_check = st.text_input("Secret Key to Check For", value="correct horse battery staple")
    text_to_check = st.text_area("Paste text here to verify...", "", height=150)

    if st.button("Detect Watermark"):
        if not text_to_check or not key_to_check:
            st.warning("Please provide text and a key.")
        else:
            with st.spinner("Analyzing text..."):
                detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           gamma=0.25,
                                           seeding_scheme="simple_1",
                                           device=model.device,
                                           tokenizer=tokenizer,
                                           hash_key=hash(key_to_check))
                
                output_dict = detector.detect(text_to_check)

                st.subheader("Detection Results")
                if output_dict.get("errors"):
                    st.error(f"An error occurred: {output_dict['errors'][0]}")
                else:
                    z_score = output_dict['z_score']
                    num_tokens = output_dict['num_tokens_scored']
                    st.metric(label="Z-Score (Threshold is ~4.0)", value=f"{z_score:.4f}")
                    st.write(f"Analyzed {num_tokens} tokens.")

                    if output_dict['prediction']:
                        st.success("**VERDICT: WATERMARK DETECTED!**")
                    else:
                        st.error("**VERDICT: WATERMARK NOT DETECTED.**")
