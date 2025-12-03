import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from pathlib import Path
import os
import gdown
import textwrap
import re

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = Path("models")
MODEL_FILENAME = "llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
DRIVE_FILE_ID = "1ldSQ7-MSj5spC8qEZp9eNoKcnE4ASfJY"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.01
MAX_TOKENS_CAP = 1024  # safety cap for token budget
MAX_CONTINUATIONS = 3  # number of times to auto-continue if model cuts off

# -------------------------------
# Ensure model exists (download if missing)
# -------------------------------
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not MODEL_PATH.exists():
        with st.spinner("Downloading LLaMA model from Google Drive (first time only)..."):
            try:
                gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return False

        if MODEL_PATH.exists():
            st.success("Model downloaded successfully.")
            return True
        else:
            st.error("Model download did not complete — file not found after download.")
            return False
    return True

# -------------------------------
# Load & cache the CTransformers LLM
# -------------------------------
@st.cache_resource
def load_llama_llm(max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=DEFAULT_TEMPERATURE):
    ok = download_model()
    if not ok:
        return None

    try:
        llm = CTransformers(
            model=str(MODEL_PATH),
            model_type='llama',
            config={
                'max_new_tokens': max_new_tokens,
                'temperature': temperature
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLaMA model: {e}")
        return None

# -------------------------------
# Utilities for ensuring complete output
# -------------------------------
def estimate_max_tokens_for_words(word_count: int) -> int:
    # heuristic: approx 1.4 tokens per word + margin
    estimated = int(word_count * 1.4) + 120
    if estimated < DEFAULT_MAX_NEW_TOKENS:
        estimated = DEFAULT_MAX_NEW_TOKENS
    if estimated > MAX_TOKENS_CAP:
        estimated = MAX_TOKENS_CAP
    return estimated

def ends_suddenly(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    # If last char is not terminal punctuation, consider abrupt end
    return text[-1] not in ('.', '!', '?')

def collapse_blank_lines(text: str) -> str:
    text = re.sub(r'(?m)^[ \t]+', '', text)        # remove leading tabs/spaces on lines
    text = re.sub(r'\n{3,}', '\n\n', text)         # collapse many blank lines
    return text.strip()

def normalize_response(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    # Remove leading indentation
    text = re.sub(r'(?m)^[ \t]+', '', text)
    # Remove heading markers
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    # Remove trailing sign-offs
    text = re.sub(r'(?i)(\n|\A)\s*(good luck|best regards|thanks|thank you|regards)[\.\!]*\s*$', '', text).strip()
    # Collapse blank lines
    text = collapse_blank_lines(text)
    # Ensure short paragraphs with commas get a punctuation mark
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    final_paragraphs = []
    for p in paragraphs:
        if p.count(".") == 0 and len(p.split()) < 40 and re.search(r'[,;:]', p):
            p = p.rstrip(" ,;:") + "."
        final_paragraphs.append(p)
    return "\n\n".join(final_paragraphs).strip()

# -------------------------------
# Strict template to request only the article
# -------------------------------
STRICT_TEMPLATE = textwrap.dedent(""" You are a professional blog writer.

Write a clear, engaging, and informative blog in around {no_words} words for a {blog_style} audience on the topic: "{input_text}".

Start the article immediately on the first line of the response. End with the last sentence of the article — nothing else. 

Do NOT mention the topic explicitly as given by the user. Do NOT repeat, restate, rephrase, or reference the prompt text in ANY form. Do NOT start with phrases like "Here is your prompt", "You asked", "As requested", etc.

Avoid labeling sections like intro/body/conclusion. Do not return any topics, explanations, outlines, headings, lists, step-by-step items, bullet points, numbered points, or meta commentary.

Make it a natural, flowing article with a smooth structure and human tone. Keep the language simple and avoid technical jargon unless necessary. """)

# -------------------------------
# Generate with auto-continue if truncated
# -------------------------------
def getLLamaresponse(llm, input_text, no_words, blog_style):
    """
    Returns a full, cleaned article text. If LLM output ends mid-sentence,
    will request continuation up to MAX_CONTINUATIONS times.
    """
    if llm is None:
        return "LLaMA model not loaded."

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=STRICT_TEMPLATE
    )

    final_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ).strip()

    try:
        raw = llm.invoke(final_prompt)
    except Exception as e:
        return f"Error generating response: {e}"

    article = normalize_response(raw)

    # If article ends abruptly, try to continue
    attempts = 0
    while attempts < MAX_CONTINUATIONS and ends_suddenly(article):
        attempts += 1
        continuation_prompt = (
            f"{article}\n\nContinue the article above. Do NOT add any headings, lists, metadata, or sign-offs. "
            "Continue directly from where the article left off and finish the article in clear paragraphs."
        )
        try:
            cont_raw = llm.invoke(continuation_prompt)
        except Exception:
            break
        if not cont_raw:
            break
        # Append continuation and re-normalize
        article = article.rstrip() + "\n\n" + cont_raw
        article = normalize_response(article)
        # Stop early if it no longer ends abruptly
        if not ends_suddenly(article):
            break

    # Final safety: if still ends abruptly, append a period
    article = article.strip()
    if article and article[-1] not in ".!?":
        article = article.rstrip() + "."

    return article

# -------------------------------
# Streamlit UI (keeps same layout and styles)
# -------------------------------
st.set_page_config(
    page_title="Short Blog Generator",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg,#6a11cb 0%, #2575fc 100%);
        padding: 28px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 30px rgba(37,117,252,0.12);
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-size: 28px; font-weight: 700; }
    .hero p { margin: 6px 0 0 0; opacity: 0.95; }
    .panel {
        background: #0b0f14;
        border-radius: 10px;
        padding: 20px;
        color: #e6eef8;
        max-width: 900px;
        margin: 0 auto 20px auto;
    }
    .card {
        background: #0b0f14;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }
    input, .stButton>button, textarea { background: #111318 !important; color: #e6eef8 !important; border-radius: 8px; }
    .muted { color: #9aa4b2; }
    .output-area {
        white-space: pre-wrap;
        line-height: 1.7;
        color: #dbeafe;
        font-size: 15px;
        padding: 10px;
        border-radius: 8px;
        background: #071021;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="hero"><h1>Short Blog Generator</h1><p>Create crisp, readable blog posts tailored to any audience.</p></div>', unsafe_allow_html=True)

with st.container():
    st.subheader("Blog input")
    input_text = st.text_input("Enter the blog topic", placeholder="e.g., The Future of Electric Vehicles")

    col1, col2 = st.columns([1, 1])
    with col1:
        no_words = st.number_input('Approximate word count', min_value=50, max_value=200, step=25, value=100)
    with col2:
        blog_style = st.selectbox('Target audience', ['Researchers', 'Data Scientist', 'Common People'])
    st.markdown("")
    submit = st.button("Generate", key="generate_button")

# -------------------------------
# On submit: ensure LLM has enough token budget and generate full article
# -------------------------------
if submit:
    if input_text.strip() == "":
        st.warning("Please enter a topic to generate your blog.")
    else:
        # Estimate needed max tokens and load llm with that config
        desired_tokens = estimate_max_tokens_for_words(no_words)
        # load a cached llm instance configured with calculated tokens and default temperature
        llm_instance = load_llama_llm(max_new_tokens=desired_tokens, temperature=DEFAULT_TEMPERATURE)

        if llm_instance is None:
            st.error("Model failed to initialize. Check model file or logs.")
        else:
            with st.spinner("Generating blog..."):
                response = getLLamaresponse(llm_instance, input_text, no_words, blog_style)

            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.subheader("Your AI-Generated Blog")

            # Convert paragraphs into safe HTML blocks (escape < and >)
            paragraphs = [p.replace("<", "&lt;").replace(">", "&gt;") for p in response.split("\n\n") if p.strip()]
            html_response = "".join(f"<p style='margin-bottom:10px'>{p}</p>" for p in paragraphs)

            st.markdown(f"<div class='output-area'>{html_response}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.success("Blog created.")
