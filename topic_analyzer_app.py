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
import os # Used for NLTK download path configuration

# --- NLTK Data Download ---
# Configure NLTK data path to be relative to the script for better portability
script_dir = os.path.dirname(__file__)
nltk_data_path = os.path.join(script_dir, "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Function to download NLTK data if missing
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

# --- Helper Functions ---
@st.cache_data # Cache preprocessing results for efficiency
def preprocess_text(text):
    """Cleans and tokenizes the input text."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers (optional, can be commented out if numbers are relevant)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    # Ensure stopwords are loaded from the correct path
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2 # Keep words longer than 2 chars
    ]
    # Return both list of tokens and a single string of processed text
    return filtered_tokens, " ".join(filtered_tokens)

@st.cache_data # Cache plotting data generation
def get_word_frequency_df(tokens, top_n=20):
    """Calculates word frequencies and returns a DataFrame."""
    if not tokens:
        return pd.DataFrame(columns=['Word', 'Frequency'])
    fdist = FreqDist(tokens)
    common_words = fdist.most_common(top_n)
    df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    return df_freq

def plot_word_frequency(df_freq):
    """Plots the word frequency bar chart."""
    if df_freq.empty:
        st.warning("No words to plot after processing.")
        return
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size
    sns.barplot(x='Frequency', y='Word', data=df_freq, ax=ax, palette='viridis', hue='Word', dodge=False, legend=False)
    ax.set_title(f'Top {len(df_freq)} Most Frequent Words')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout() # Adjust layout
    st.pyplot(fig)

@st.cache_data # Cache word cloud generation
def generate_word_cloud_image(processed_text_string):
    """Generates and displays a word cloud."""
    if not processed_text_string.strip():
        # Return None or a placeholder if no text
        return None
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(processed_text_string)
        return wordcloud
    except ValueError: # Handle cases where text might be too short or only contains ignored words
         return None

def display_word_cloud(wordcloud_obj):
    """Displays the generated word cloud object."""
    if wordcloud_obj:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_obj, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Could not generate word cloud. Input text might be too short or lack meaningful words after processing.")


@st.cache_data # Cache topic modeling results
def get_main_topic(processed_text_string, n_topics=1, n_top_words=10):
    """Identifies main topics using LDA."""
    if not processed_text_string.strip():
         return None, "Input text is empty after processing."

    # Use CountVectorizer for LDA
    # Consider adding max_features to limit vocabulary size if needed
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        # Pass as a list/iterable
        tf = vectorizer.fit_transform([processed_text_string])
        # Check if vocabulary is empty after vectorization
        feature_names = vectorizer.get_feature_names_out()
        if not feature_names.any():
             return None, "Not enough meaningful words found to determine a topic after filtering (min_df=2)."
        # Check if the document-term matrix is empty or has insufficient terms for LDA
        if tf.shape[1] < n_topics:
             return None, f"Insufficient unique terms ({tf.shape[1]}) to form {n_topics} topic(s)."

    except ValueError:
        # This might catch errors if the input is completely empty or unprocessable
        return None, "Text vectorization failed. Input might be unsuitable."

    # Fit LDA model
    # Using n_init to improve stability, though it increases computation time
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_init=10)
    try:
        lda.fit(tf)
    except Exception as e:
        return None, f"LDA model fitting failed: {e}"


    topics_list = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        # Ensure we don't request more words than available features
        actual_top_words = min(n_top_words, len(feature_names))
        top_words_indices = topic_weights.argsort()[:-actual_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_list.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    if not topics_list:
        return None, "Could not extract topics from the LDA model."

    return topics_list, None # Return list of topics and no error message


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Text Analyzer")
st.title("📝 Sophisticated Text Analyzer")
st.write("""
Enter text below to generate insights:
- **Word Frequency Plot:** Visualize the most common words (excluding common stop words).
- **Word Cloud:** A visual representation of word prominence.
- **Main Topic:** An attempt to identify the primary topic using LDA.
""")

# Download NLTK data first
download_nltk_data()

# Input Text Area
input_text = st.text_area("Input Text:", height=250, placeholder="Paste or type your text here...")

# Analyze Button
analyze_button = st.button("Analyze Text")

if analyze_button and input_text:
    st.markdown("---")
    st.subheader("Analysis Results")

    # 1. Preprocess Text
    with st.spinner("Processing text..."):
        tokens, processed_text_string = preprocess_text(input_text)

    if not tokens:
        st.warning("The input text seems to contain only stopwords, punctuation, or numbers. Please provide more substantial text.")
    else:
        # Use columns for layout
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
                 # Use the processed string for LDA
                 topics, error_msg = get_main_topic(processed_text_string, n_topics=1, n_top_words=10) # Requesting 1 main topic
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


elif analyze_button and not input_text:
    st.warning("⚠️ Please enter some text into the text area to analyze.")

st.markdown("---")
st.caption("Powered by Streamlit and Python Libraries")
