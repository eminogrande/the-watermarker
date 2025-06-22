import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_model():
    """Loads and caches a GPT-2 model and tokenizer from Hugging Face."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def get_vocab(tokenizer):
    """Pulls the vocabulary from the loaded tokenizer."""
    vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    return vocab

# --- Core Application Logic ---
st.set_page_config(layout="wide", page_title="Real LLM Watermarker")
st.title("Real Probabilistic Watermarking Tool (GPT-2 Edition)")

# --- Load the Model ---
with st.spinner("Loading GPT-2 model..."):
    model, tokenizer = load_model()
    # Set a padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

st.success("Model loaded and ready!")

# Extract vocabulary for the watermarker
model_vocab = get_vocab(tokenizer)

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
                watermark_processor = WatermarkLogitsProcessor(vocab=model_vocab,
                                                             gamma=0.25,
                                                             delta=2.0,
                                                             seeding_scheme="simple_1",
                                                             hash_key=hash(secret_key))
                
                prompt = f"Paraphrase the following text: {text_to_paraphrase}\n\nParaphrased text:"
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Generate text using the model and the watermark processor
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(len(text_to_paraphrase.split()) * 1.5),
                    logits_processor=[watermark_processor],
                    temperature=0.7,
                    top_k=50,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode the output, skipping special tokens
                watermarked_text = tokenizer.decode(output[0], skip_special_tokens=True)
                # Clean up the output to only show the paraphrased part
                watermarked_text = watermarked_text.split("Paraphrased text:")[1].strip()

                st.session_state['generated_text'] = watermarked_text

    if 'generated_text' in st.session_state:
        st.subheader("Your Paraphrased and Watermarked Text:")
        st.text_area("Output Text", st.session_state.generated_text, height=200)


with col2:
    st.header("2. Detect Watermark")
    key_to_check = st.text_input("Secret Key to Check For", value="correct horse battery staple")
    text_to_check_default = st.session_state.get('generated_text', '')
    text_to_check = st.text_area("Paste text here to verify...", text_to_check_default, height=150)

    if st.button("Detect Watermark"):
        if not text_to_check or not key_to_check:
            st.warning("Please provide text and a key.")
        else:
            with st.spinner("Analyzing text..."):
                # The detector needs the original tokenizer
                detector = WatermarkDetector(vocab=model_vocab,
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
                    z_score = output_dict.get('z_score', 0)
                    num_tokens = output_dict.get('num_tokens_scored', 0)
                    st.metric(label="Z-Score (Threshold is ~4.0)", value=f"{z_score:.4f}")
                    st.write(f"Analyzed {num_tokens} tokens.")

                    if output_dict.get('prediction'):
                        st.success("**VERDICT: WATERMARK DETECTED!**")
                    else:
                        st.error("**VERDICT: WATERMARK NOT DETECTED.**")
