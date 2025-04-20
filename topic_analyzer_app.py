import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import os

# --- Image Processing Imports ---
from PIL import Image
import numpy as np
# Ensure TensorFlow is installed: pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow/Keras is not installed. Image analysis features will be disabled. Install using: pip install tensorflow")

# --- NLTK Data Download Configuration (same as before) ---
script_dir = os.path.dirname(__file__)
nltk_data_path = os.path.join(script_dir, "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords', paths=[nltk_data_path])
    except LookupError:
        st.info("Downloading 'stopwords' corpus for NLTK...")
        nltk.download('stopwords', download_dir=nltk_data_path)
        st.success("'stopwords' downloaded.")
    try:
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_path])
    except LookupError:
        st.info("Downloading 'punkt' tokenizer for NLTK...")
        nltk.download('punkt', download_dir=nltk_data_path)
        st.success("'punkt' downloaded.")

# --- Text Processing Functions (mostly same as before) ---
@st.cache_data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return filtered_tokens, " ".join(filtered_tokens)

@st.cache_data
def get_word_frequency_df(tokens, top_n=20):
    if not tokens:
        return pd.DataFrame(columns=['Word', 'Frequency'])
    fdist = FreqDist(tokens)
    common_words = fdist.most_common(top_n)
    df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    return df_freq

def plot_word_frequency(df_freq):
    if df_freq.empty:
        st.warning("No words to plot after processing.")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Frequency', y='Word', data=df_freq, ax=ax, palette='viridis', hue='Word', dodge=False, legend=False)
    ax.set_title(f'Top {len(df_freq)} Most Frequent Words')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def generate_word_cloud_image(processed_text_string):
    if not processed_text_string.strip():
        return None
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(processed_text_string)
        return wordcloud
    except ValueError:
         return None

def display_word_cloud(wordcloud_obj):
    if wordcloud_obj:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_obj, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Could not generate word cloud.")

@st.cache_data
def get_main_topic(processed_text_string, n_topics=1, n_top_words=10):
    if not processed_text_string.strip():
         return None, "Input text is empty after processing."
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        tf = vectorizer.fit_transform([processed_text_string])
        feature_names = vectorizer.get_feature_names_out()
        if not feature_names.any():
             return None, "Not enough meaningful words found to determine a topic after filtering (min_df=2)."
        if tf.shape[1] < n_topics:
             return None, f"Insufficient unique terms ({tf.shape[1]}) to form {n_topics} topic(s)."
    except ValueError:
        return None, "Text vectorization failed."

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_init=10)
    try:
        lda.fit(tf)
    except Exception as e:
        return None, f"LDA model fitting failed: {e}"

    topics_list = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        actual_top_words = min(n_top_words, len(feature_names))
        top_words_indices = topic_weights.argsort()[:-actual_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_list.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    if not topics_list:
        return None, "Could not extract topics from the LDA model."
    return topics_list, None

# --- Image Processing Functions ---
@st.cache_resource # Cache the model itself
def load_vgg_model():
    """Loads the VGG16 model pre-trained on ImageNet."""
    if not TF_AVAILABLE:
        return None
    try:
        model = VGG16(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading VGG16 model: {e}")
        return None

@st.cache_data # Cache results based on image bytes
def process_and_predict_image(image_bytes, model, top_n=5):
    """Processes an image and returns top N predictions using the VGG model."""
    if not TF_AVAILABLE or model is None:
        return None, "TensorFlow or VGG model not available."

    try:
        # Load image from bytes
        img = Image.open(image_bytes).convert('RGB') # Ensure 3 channels

        # Resize and preprocess for VGG16
        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded) # Use VGG16's preprocess_input

        # Make prediction
        predictions = model.predict(img_preprocessed)
        decoded_predictions = decode_predictions(predictions, top=top_n)[0] # Get top N

        # Format results
        results_df = pd.DataFrame(decoded_predictions, columns=['id', 'label', 'probability'])
        results_df['probability'] = results_df['probability'].map('{:.2%}'.format) # Format probability

        return results_df[['label', 'probability']], None # Return DataFrame and no error
    except Exception as e:
        return None, f"Error processing or predicting image: {e}"

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Text & Image Analyzer")
st.title("📝🖼️ Multi-Modal Analyzer")
st.write("""
Select the type of input you want to analyze (Text or Image) and provide the content below.
- **Text Analysis:** Generates word frequency, word cloud, and identifies the main topic (LDA).
- **Image Analysis:** Identifies the content of the image using a VGG16 model.
""")

# Download NLTK data (only needed for text)
download_nltk_data()

# --- Input Selection ---
input_type = st.radio(
    "Select Input Type:",
    ('Text', 'Image'),
    horizontal=True,
    key='input_type_selector'
)

# --- Conditional Input Area ---
input_text = None
uploaded_image = None

if input_type == 'Text':
    input_text = st.text_area("Input Text:", height=250, placeholder="Paste or type your text here...")
    analyze_button = st.button("Analyze Text", key="analyze_text_button")

elif input_type == 'Image':
    if not TF_AVAILABLE:
        st.warning("Image analysis requires TensorFlow. Please install it.")
    else:
        uploaded_image = st.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png'])
        analyze_button = st.button("Analyze Image", key="analyze_image_button", disabled=(uploaded_image is None)) # Disable button if no image

else: # Should not happen with radio buttons
    analyze_button = False


# --- Analysis Execution ---
st.markdown("---")

if analyze_button:
    if input_type == 'Text':
        if input_text:
            st.subheader("🔎 Text Analysis Results")
            # 1. Preprocess Text
            with st.spinner("Processing text..."):
                tokens, processed_text_string = preprocess_text(input_text)

            if not tokens:
                st.warning("The input text seems to contain only stopwords or punctuation. Please provide more substantial text.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    # 2. Plot Word Frequency
                    st.subheader("📊 Word Frequency Plot")
                    with st.spinner("Generating frequency plot..."):
                        df_freq = get_word_frequency_df(tokens)
                        plot_word_frequency(df_freq)

                    # 4. Topic Modeling
                    st.subheader("💡 Main Topic(s)")
                    with st.spinner("Identifying main topic(s)..."):
                         topics, error_msg = get_main_topic(processed_text_string, n_topics=1, n_top_words=10)
                         if topics:
                             for topic in topics:
                                 st.write(topic)
                         else:
                             st.warning(f"Could not determine topic. Reason: {error_msg}")
                with col2:
                    # 3. Generate Word Cloud
                    st.subheader("☁️ Word Cloud")
                    with st.spinner("Generating word cloud..."):
                        wordcloud_obj = generate_word_cloud_image(processed_text_string)
                        display_word_cloud(wordcloud_obj)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    elif input_type == 'Image':
        if uploaded_image is not None and TF_AVAILABLE:
            st.subheader("🖼️ Image Analysis Results")
            col_img, col_pred = st.columns(2)

            with col_img:
                st.subheader("Uploaded Image")
                # Read image bytes for processing and display
                image_bytes = uploaded_image.getvalue()
                st.image(image_bytes, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)

            with col_pred:
                st.subheader("🔍 Image Content Prediction (VGG16)")
                with st.spinner("Loading VGG model and analyzing image..."):
                    # Load model (cached)
                    vgg_model = load_vgg_model()
                    if vgg_model:
                        # Process and predict (cached based on bytes)
                        predictions_df, error_msg = process_and_predict_image(image_bytes, vgg_model, top_n=5)

                        if predictions_df is not None:
                            st.dataframe(predictions_df, use_container_width=True)
                        else:
                            st.error(f"Analysis failed: {error_msg}")
                    else:
                        st.error("Could not load the VGG model.")
        elif not TF_AVAILABLE:
             st.error("Cannot analyze image because TensorFlow is not available.")
        else:
            st.warning("⚠️ Please upload an image file to analyze.")


st.markdown("---")
st.caption("Powered by Streamlit, NLTK, Scikit-learn, TensorFlow/Keras, and more!")
