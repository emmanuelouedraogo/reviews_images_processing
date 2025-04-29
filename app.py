import streamlit as st
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import string
import re
import os
import io # Added for BytesIO
from typing import List, Tuple, Optional, Dict, Any

# --- Machine Learning / NLP Imports ---
from nltk.corpus import stopwords, wordnet # Added wordnet
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer # Added Lemmatizer
from nltk import pos_tag # Added pos_tag
from nltk.corpus import words as nltk_words # Added words corpus

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- TensorFlow/Keras for Image Processing (Optional) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    # Use tf.keras.utils for newer TF versions
    try:
        from tensorflow.keras.utils import img_to_array
    except ImportError:
        from tensorflow.keras.preprocessing.image import img_to_array # Fallback for older TF
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Display error during app setup if TF is needed but missing
    # st.error("TensorFlow/Keras is not installed. Image analysis features will be disabled. Install using: pip install tensorflow")


# --- Constants ---
SCRIPT_DIR = os.path.dirname(__file__)
NLTK_DATA_PATH = os.path.join(SCRIPT_DIR, "nltk_data")
DEFAULT_TOP_N_WORDS = 20
DEFAULT_LDA_N_TOPICS = 1
DEFAULT_LDA_N_TOP_WORDS = 10
DEFAULT_MIN_WORD_LENGTH = 3 # Added constant for min word length
VGG_IMG_SIZE = (224, 224)

# --- NLTK Data Download Configuration ---

def configure_nltk_path():
    """Configures the NLTK data path and creates the directory if needed."""
    if not os.path.exists(NLTK_DATA_PATH):
        try:
            os.makedirs(NLTK_DATA_PATH)
            print(f"Created NLTK data directory: {NLTK_DATA_PATH}")
        except OSError as e:
            st.error(f"Failed to create NLTK data directory: {e}")
            return False # Indicate failure
    # Append path only if it exists and isn't already there
    if NLTK_DATA_PATH not in nltk.data.path:
         nltk.data.path.append(NLTK_DATA_PATH)
    print(f"Using NLTK data path: {nltk.data.path}") # Debug print
    return True # Indicate success

def download_nltk_data_if_missing(resource_path: str, download_name: str):
    """Downloads a specific NLTK resource if it's not found."""
    try:
        # Check if resource is available in the custom path first
        nltk.data.find(resource_path, paths=[NLTK_DATA_PATH])
        print(f"NLTK resource '{download_name}' found.")
        return True # Indicate resource is available
    except LookupError:
        st.info(f"Downloading NLTK resource '{download_name}'...")
        try:
            nltk.download(download_name, download_dir=NLTK_DATA_PATH)
            st.success(f"NLTK resource '{download_name}' downloaded successfully to {NLTK_DATA_PATH}.")
            # Verify download by trying to find it again
            nltk.data.find(resource_path, paths=[NLTK_DATA_PATH])
            return True # Indicate success
        except Exception as e:
            st.error(f"Failed to download NLTK resource '{download_name}'. Error: {e}")
            st.error("Text analysis features might not work correctly.")
            return False # Indicate failure

# --- Text Processing Functions ---

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convertit les étiquettes de parties du discours (Part of Speech, POS) de Penn Treebank en format WordNet.

    Args:
        treebank_tag (str): Étiquette de partie du discours au format Penn Treebank.

    Returns:
        str: Étiquette de partie du discours au format WordNet (e.g., wordnet.NOUN).
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        # Par défaut, considérer comme un nom si non reconnu
        return wordnet.NOUN

# Updated preprocess_text function incorporating lemmatization and POS tagging
@st.cache_data
def preprocess_text(text: str) -> Tuple[List[str], str]:
    """
    Pré-traite un texte : conversion en minuscules, suppression des caractères non
    alphabétiques, élimination des stop words, filtrage des mots non anglais,
    lemmatisation basée sur le POS tagging, et filtration des mots courts.

    Args:
        text (str): Texte à pré-traiter.

    Returns:
        Tuple[List[str], str]: Un tuple contenant :
            - La liste des tokens finaux après traitement.
            - Une chaîne de caractères contenant les tokens finaux joints par des espaces.
    """
    if not isinstance(text, str) or not text.strip():
        return [], "" # Handle non-string or empty input gracefully

    # --- Load NLTK resources needed within the function ---
    try:
        # Ensure stopwords are loaded from the correct path if NLTK setup succeeded
        stop_words = set(stopwords.words('english', paths=[NLTK_DATA_PATH]))
    except LookupError:
         # Fallback or handle error if stopwords weren't downloaded
         st.warning("Stopwords list not found. Proceeding without stopword removal.")
         stop_words = set() # Use an empty set if download failed

    try:
        # Load English words dictionary
        english_words = set(nltk_words.words(paths=[NLTK_DATA_PATH]))
    except LookupError:
        st.warning("NLTK 'words' corpus not found. Cannot filter non-English words.")
        english_words = None # Flag that filtering cannot be done

    # --- Start Preprocessing ---
    text = text.lower()
    # Remove non-alphabetic characters (keeps spaces)
    text = re.sub("[^a-z\s]", "", text) # Keep spaces for splitting

    # Tokenize (simple split after regex cleaning)
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Filter for English words if the corpus was loaded
    if english_words is not None:
        words = [word for word in words if word in english_words]

    if not words:
        return [], "" # Return empty if no words remain after filtering

    # POS Tagging (Requires 'averaged_perceptron_tagger')
    try:
        tagged_words = pos_tag(words)
    except LookupError:
        st.error("NLTK 'averaged_perceptron_tagger' not found. Cannot perform lemmatization.")
        # Fallback: return words after stopword/English filtering but before lemmatization
        final_tokens = [w for w in words if len(w) >= DEFAULT_MIN_WORD_LENGTH]
        processed_text_string = " ".join(final_tokens)
        return final_tokens, processed_text_string
    except Exception as e:
        st.error(f"Error during POS tagging: {e}")
        return [], "" # Return empty on unexpected error


    # Lemmatization (Requires 'wordnet')
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatized_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tagged_words
        ]
    except LookupError:
        # This might happen if wordnet failed download despite earlier checks
        st.error("NLTK 'wordnet' resource not found during lemmatization.")
        # Fallback: return tagged words before lemmatization attempt
        final_tokens = [w for w in words if len(w) >= DEFAULT_MIN_WORD_LENGTH]
        processed_text_string = " ".join(final_tokens)
        return final_tokens, processed_text_string
    except Exception as e:
        st.error(f"Error during lemmatization: {e}")
        return [], "" # Return empty on unexpected error


    # Filter by minimum length
    final_tokens = [w for w in lemmatized_words if len(w) >= DEFAULT_MIN_WORD_LENGTH]

    # Prepare the joined string output
    processed_text_string = " ".join(final_tokens)

    return final_tokens, processed_text_string


@st.cache_data
def get_word_frequency_df(tokens: List[str], top_n: int = DEFAULT_TOP_N_WORDS) -> pd.DataFrame:
    """
    Calculates word frequencies and returns a DataFrame.

    Args:
        tokens: A list of word tokens.
        top_n: The number of most frequent words to return.

    Returns:
        A pandas DataFrame with 'Word' and 'Frequency' columns.
    """
    if not tokens:
        return pd.DataFrame(columns=['Word', 'Frequency'])
    fdist = FreqDist(tokens)
    common_words = fdist.most_common(top_n)
    df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    return df_freq

def plot_word_frequency(df_freq: pd.DataFrame):
    """Plots the word frequency bar chart using Matplotlib/Seaborn."""
    if df_freq.empty:
        st.warning("No words to plot after processing.")
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_freq) * 0.4))) # Dynamic height
    sns.barplot(x='Frequency', y='Word', data=df_freq, ax=ax, palette='viridis', hue='Word', dodge=False, legend=False)
    ax.set_title(f'Top {len(df_freq)} Most Frequent Words')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

@st.cache_data
def generate_word_cloud_image(processed_text_string: str) -> Optional[WordCloud]:
    """
    Generates a WordCloud object from processed text.

    Args:
        processed_text_string: A single string of space-separated processed words.

    Returns:
        A WordCloud object or None if generation fails or text is empty.
    """
    if not processed_text_string.strip():
        return None
    try:
        # Added max_words for performance/visuals on very large texts
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='viridis', max_words=200).generate(processed_text_string)
        return wordcloud
    except ValueError as e:
         # Handle cases where text might be too short or only contains ignored words
         print(f"WordCloud generation error: {e}") # Log error for debugging
         st.warning(f"Could not generate word cloud: {e}") # Show warning in UI
         return None

def display_word_cloud(wordcloud_obj: Optional[WordCloud]):
    """Displays the generated word cloud object using Matplotlib."""
    if wordcloud_obj:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_obj, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig) # Close the figure
    else:
        # Warning is now generated in generate_word_cloud_image if needed
        pass
        # st.warning("Could not generate word cloud. Input text might be too short or lack meaningful words after processing.")

@st.cache_data
def get_main_topics(processed_text_string: str, n_topics: int = DEFAULT_LDA_N_TOPICS, n_top_words: int = DEFAULT_LDA_N_TOP_WORDS) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Identifies main topics using Latent Dirichlet Allocation (LDA).

    Args:
        processed_text_string: A single string of space-separated processed words.
        n_topics: The number of topics to identify.
        n_top_words: The number of top words to display for each topic.

    Returns:
        A tuple containing:
            - A list of strings, each describing a topic, or None on failure.
            - An error message string if failed, otherwise None.
    """
    if not processed_text_string.strip():
         return None, "Input text is empty after processing."

    # Use CountVectorizer for LDA
    # Note: CountVectorizer has its own stop_words='english'. This might be redundant
    # with our preprocessing but is generally safe.
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        # Pass as a list/iterable
        tf = vectorizer.fit_transform([processed_text_string])
        feature_names = vectorizer.get_feature_names_out()

        # Check vocabulary and term count *after* fitting
        if not feature_names.any():
             return None, "Vectorization resulted in an empty vocabulary (min_df=2 might be too high, or text lacks diverse words after processing)."
        # Ensure enough features for the requested number of topics for LDA
        if tf.shape[1] < n_topics:
             return None, f"Insufficient unique terms ({tf.shape[1]}) after vectorization to form {n_topics} topic(s). Try reducing the number of topics."

    except ValueError as e:
        # This might catch errors if the input is completely empty or unprocessable by CountVectorizer
        return None, f"Text vectorization failed. Input might be unsuitable. Error: {e}"

    # Fit LDA model
    # Using n_init for stability, consider adjusting based on performance needs
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_init=10, learning_method='online') # 'online' can be faster
    try:
        lda.fit(tf)
    except ValueError as e:
         # Catch potential issues during LDA fitting (e.g., insufficient features for components)
         return None, f"LDA model fitting failed. Check if the number of topics ({n_topics}) is feasible for the processed text. Error: {e}"
    except Exception as e:
        # Catch other potential issues during LDA fitting
        return None, f"LDA model fitting failed. Error: {e}"


    topics_list = []
    if not hasattr(lda, 'components_') or lda.components_ is None:
         return None, "LDA model did not produce components."

    for topic_idx, topic_weights in enumerate(lda.components_):
        # Ensure we don't request more words than available features
        actual_top_words = min(n_top_words, len(feature_names))
        # Get indices of top words for this topic
        top_words_indices = topic_weights.argsort()[:-actual_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_list.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    if not topics_list:
        # This case might occur if LDA runs but fails to extract meaningful topics
        return None, "Could not extract meaningful topics from the LDA model components."

    return topics_list, None # Return list of topics and no error message

# --- Image Processing Functions ---
@st.cache_resource # Cache the model itself for the duration of the session
def load_vgg_model() -> Optional[Any]: # Using Any as TF types can be complex
    """Loads the VGG16 model pre-trained on ImageNet."""
    if not TF_AVAILABLE:
        st.error("TensorFlow is required for image analysis but not installed.")
        return None
    try:
        with st.spinner("Loading VGG16 model (this may take a moment)..."):
            model = VGG16(weights='imagenet')
        st.success("VGG16 model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading VGG16 model: {e}")
        return None

@st.cache_data # Cache results based on image bytes and model hash
def process_and_predict_image(image_bytes: bytes, _model_ref: Any, top_n: int = 5) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Processes an image and returns top N predictions using the VGG model.

    Args:
        image_bytes: The raw bytes of the image file.
        _model_ref: Reference to the loaded model (used for caching invalidation).
        top_n: The number of top predictions to return.

    Returns:
        A tuple containing:
            - A pandas DataFrame with 'label' and 'probability' columns, or None on failure.
            - An error message string if failed, otherwise None.
    """
    # Model is loaded outside and passed in, but check TF availability again
    if not TF_AVAILABLE:
        return None, "TensorFlow is not available."
    if _model_ref is None:
         return None, "VGG model is not loaded."

    model = _model_ref # Use the actual model object

    try:
        # Load image from bytes using PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure 3 channels

        # Resize and preprocess for VGG16
        img_resized = img.resize(VGG_IMG_SIZE, Image.Resampling.LANCZOS) # Use high-quality resampling
        img_array = img_to_array(img_resized) # Use the imported function
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded) # VGG16 specific preprocessing

        # Make prediction
        predictions = model.predict(img_preprocessed)
        # Decode predictions using ImageNet labels
        decoded_predictions = decode_predictions(predictions, top=top_n)[0] # Get top N results

        # Format results into a DataFrame
        results_list = [(label, f"{prob:.2%}") for _, label, prob in decoded_predictions]
        results_df = pd.DataFrame(results_list, columns=['Label', 'Probability'])

        return results_df, None # Return DataFrame and no error

    except Exception as e:
        # Catch potential errors during image loading, processing, or prediction
        st.error(f"Error processing or predicting image: {e}") # Show error in UI
        return None, f"Error processing or predicting image: {e}"


# --- Streamlit App Main Function ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Text & Image Analyzer")
    st.title("📝🖼️ Multi-Modal Analyzer")
    st.write("""
    Analyze text to find frequent words, generate word clouds, and identify topics (now with lemmatization!),
    or analyze images to predict their content using a pre-trained model.
    """)
    st.sidebar.header("Configuration")

    # --- NLTK Setup ---
    if not configure_nltk_path():
         st.stop() # Stop execution if NLTK path setup fails

    # Download necessary NLTK data (only if needed for text analysis)
    # These checks run once per session if successful due to caching/app state
    # Use resource paths for checking, download names for downloading
    nltk_resources_ok = []
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/stopwords', 'stopwords'))
    nltk_resources_ok.append(download_nltk_data_if_missing('tokenizers/punkt', 'punkt'))
    # Add downloads for the new preprocessing steps
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/wordnet', 'wordnet'))
    nltk_resources_ok.append(download_nltk_data_if_missing('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'))
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/words', 'words'))


    if not all(nltk_resources_ok):
         st.warning("One or more NLTK resources failed to download. Text analysis might be affected or incomplete.")
         # Depending on which failed, specific features might not work.
         # The preprocess_text function now has internal checks for some of these.


    # --- Input Selection ---
    st.sidebar.markdown("---")
    input_options = ['Text']
    if TF_AVAILABLE:
        input_options.append('Image')
    else:
        st.sidebar.warning("Image analysis disabled: TensorFlow not found.")

    input_type = st.sidebar.radio(
        "Select Input Type:",
        options=input_options,
        horizontal=True,
        key='input_type_selector'
    )

    # --- Analysis Parameters (Sidebar) ---
    st.sidebar.markdown("---")
    if input_type == 'Text':
        st.sidebar.subheader("Text Analysis Options")
        top_n_freq = st.sidebar.slider("Top N Frequent Words:", min_value=5, max_value=50, value=DEFAULT_TOP_N_WORDS, step=1)
        lda_n_topics = st.sidebar.number_input("Number of Topics (LDA):", min_value=1, max_value=10, value=DEFAULT_LDA_N_TOPICS, step=1)
        lda_n_words = st.sidebar.slider("Words per Topic (LDA):", min_value=3, max_value=20, value=DEFAULT_LDA_N_TOP_WORDS, step=1)
        # Note: min_word_length is now fixed in preprocess_text, could be made configurable here too

    elif input_type == 'Image':
        st.sidebar.subheader("Image Analysis Options")
        top_n_pred = st.sidebar.slider("Top N Predictions:", min_value=1, max_value=10, value=5, step=1)
        # Load the model once when Image analysis is selected
        vgg_model = load_vgg_model()


    # --- Main Panel for Input and Analysis ---
    st.markdown("---") # Separator in main panel

    analyze_button_pressed = False
    input_text = None
    uploaded_image = None

    if input_type == 'Text':
        input_text = st.text_area("Input Text:", height=250, placeholder="Paste or type your text here...")
        if st.button("Analyze Text", key="analyze_text_button"):
            if input_text and input_text.strip(): # Check if text is not just whitespace
                analyze_button_pressed = True
            else:
                st.warning("⚠️ Please enter some text to analyze.")

    elif input_type == 'Image' and TF_AVAILABLE:
        uploaded_image = st.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png'])
        # Disable button if no image is uploaded or model failed to load
        disable_image_button = (uploaded_image is None) or (vgg_model is None)
        if st.button("Analyze Image", key="analyze_image_button", disabled=disable_image_button):
             if uploaded_image is not None and vgg_model is not None:
                 analyze_button_pressed = True
             elif vgg_model is None:
                  st.error("Cannot analyze image: VGG model failed to load.")
             else:
                 st.warning("⚠️ Please upload an image file to analyze.")

    # --- Analysis Execution ---
    if analyze_button_pressed:
        st.markdown("---")
        st.subheader(f"🔎 {input_type} Analysis Results")

        if input_type == 'Text':
            with st.spinner("Processing text (including lemmatization)..."):
                # The new preprocess_text is called here
                tokens, processed_text_string = preprocess_text(input_text)

            if not tokens:
                st.warning("The input text resulted in no valid words after processing (check for only stopwords, non-English words, etc.).")
            else:
                # Use columns for better layout
                col1, col2 = st.columns(2)
                with col1:
                    # Word Frequency
                    st.markdown("##### 📊 Word Frequency Plot")
                    with st.spinner("Generating frequency plot..."):
                        df_freq = get_word_frequency_df(tokens, top_n=top_n_freq)
                        plot_word_frequency(df_freq)

                    # Topic Modeling
                    st.markdown("##### 💡 Main Topic(s) (LDA)")
                    with st.spinner("Identifying main topic(s)..."):
                         topics, error_msg = get_main_topics(processed_text_string, n_topics=lda_n_topics, n_top_words=lda_n_words)
                         if topics:
                             for topic in topics:
                                 st.write(topic)
                         else:
                             st.warning(f"Could not determine topic(s). Reason: {error_msg}")
                with col2:
                    # Word Cloud
                    st.markdown("##### ☁️ Word Cloud")
                    with st.spinner("Generating word cloud..."):
                        wordcloud_obj = generate_word_cloud_image(processed_text_string)
                        display_word_cloud(wordcloud_obj) # Handles None case internally

        elif input_type == 'Image' and uploaded_image is not None and vgg_model is not None:
            col_img, col_pred = st.columns([0.6, 0.4]) # Adjust column widths

            with col_img:
                st.markdown("##### Uploaded Image")
                image_bytes = uploaded_image.getvalue()
                try:
                    st.image(image_bytes, caption=f"Uploaded: {uploaded_image.name}", use_column_width='always')
                except Exception as e:
                    st.error(f"Could not display the uploaded image. Error: {e}")


            with col_pred:
                st.markdown(f"##### 🔍 Image Content Prediction (Top {top_n_pred})")
                with st.spinner("Analyzing image..."):
                    # Pass model reference for caching check
                    predictions_df, error_msg = process_and_predict_image(image_bytes, vgg_model, top_n=top_n_pred)

                    if predictions_df is not None:
                        # Use st.dataframe for better table display
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                    else:
                        # Error message already shown in process_and_predict_image
                        st.error(f"Image analysis failed. {error_msg or ''}")

    st.markdown("---")
    st.caption("Powered by Streamlit, NLTK, Scikit-learn, Matplotlib, Seaborn, WordCloud" + (", TensorFlow/Keras" if TF_AVAILABLE else ""))

if __name__ == "__main__":
    # Suggest creating a requirements file if it doesn't exist
    # print("Consider creating a requirements.txt file for dependencies:")
    # print("pip freeze > requirements.txt")
    main()
import streamlit as st
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import string
import re
import os
import io # Added for BytesIO
from typing import List, Tuple, Optional, Dict, Any

# --- Machine Learning / NLP Imports ---
from nltk.corpus import stopwords, wordnet # Added wordnet
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer # Added Lemmatizer
from nltk import pos_tag # Added pos_tag
from nltk.corpus import words as nltk_words # Added words corpus

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- TensorFlow/Keras for Image Processing (Optional) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    # Use tf.keras.utils for newer TF versions
    try:
        from tensorflow.keras.utils import img_to_array
    except ImportError:
        from tensorflow.keras.preprocessing.image import img_to_array # Fallback for older TF
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Display error during app setup if TF is needed but missing
    # st.error("TensorFlow/Keras is not installed. Image analysis features will be disabled. Install using: pip install tensorflow")


# --- Constants ---
SCRIPT_DIR = os.path.dirname(__file__)
NLTK_DATA_PATH = os.path.join(SCRIPT_DIR, "nltk_data")
DEFAULT_TOP_N_WORDS = 20
DEFAULT_LDA_N_TOPICS = 1
DEFAULT_LDA_N_TOP_WORDS = 10
DEFAULT_MIN_WORD_LENGTH = 3 # Added constant for min word length
VGG_IMG_SIZE = (224, 224)

# --- NLTK Data Download Configuration ---

def configure_nltk_path():
    """Configures the NLTK data path and creates the directory if needed."""
    if not os.path.exists(NLTK_DATA_PATH):
        try:
            os.makedirs(NLTK_DATA_PATH)
            print(f"Created NLTK data directory: {NLTK_DATA_PATH}")
        except OSError as e:
            st.error(f"Failed to create NLTK data directory: {e}")
            return False # Indicate failure
    # Append path only if it exists and isn't already there
    if NLTK_DATA_PATH not in nltk.data.path:
         nltk.data.path.append(NLTK_DATA_PATH)
    print(f"Using NLTK data path: {nltk.data.path}") # Debug print
    return True # Indicate success

def download_nltk_data_if_missing(resource_path: str, download_name: str):
    """Downloads a specific NLTK resource if it's not found."""
    try:
        # Check if resource is available in the custom path first
        nltk.data.find(resource_path, paths=[NLTK_DATA_PATH])
        print(f"NLTK resource '{download_name}' found.")
        return True # Indicate resource is available
    except LookupError:
        st.info(f"Downloading NLTK resource '{download_name}'...")
        try:
            nltk.download(download_name, download_dir=NLTK_DATA_PATH)
            st.success(f"NLTK resource '{download_name}' downloaded successfully to {NLTK_DATA_PATH}.")
            # Verify download by trying to find it again
            nltk.data.find(resource_path, paths=[NLTK_DATA_PATH])
            return True # Indicate success
        except Exception as e:
            st.error(f"Failed to download NLTK resource '{download_name}'. Error: {e}")
            st.error("Text analysis features might not work correctly.")
            return False # Indicate failure

# --- Text Processing Functions ---

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convertit les étiquettes de parties du discours (Part of Speech, POS) de Penn Treebank en format WordNet.

    Args:
        treebank_tag (str): Étiquette de partie du discours au format Penn Treebank.

    Returns:
        str: Étiquette de partie du discours au format WordNet (e.g., wordnet.NOUN).
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        # Par défaut, considérer comme un nom si non reconnu
        return wordnet.NOUN

# Updated preprocess_text function incorporating lemmatization and POS tagging
@st.cache_data
def preprocess_text(text: str) -> Tuple[List[str], str]:
    """
    Pré-traite un texte : conversion en minuscules, suppression des caractères non
    alphabétiques, élimination des stop words, filtrage des mots non anglais,
    lemmatisation basée sur le POS tagging, et filtration des mots courts.

    Args:
        text (str): Texte à pré-traiter.

    Returns:
        Tuple[List[str], str]: Un tuple contenant :
            - La liste des tokens finaux après traitement.
            - Une chaîne de caractères contenant les tokens finaux joints par des espaces.
    """
    if not isinstance(text, str) or not text.strip():
        return [], "" # Handle non-string or empty input gracefully

    # --- Load NLTK resources needed within the function ---
    try:
        # Ensure stopwords are loaded from the correct path if NLTK setup succeeded
        stop_words = set(stopwords.words('english', paths=[NLTK_DATA_PATH]))
    except LookupError:
         # Fallback or handle error if stopwords weren't downloaded
         st.warning("Stopwords list not found. Proceeding without stopword removal.")
         stop_words = set() # Use an empty set if download failed

    try:
        # Load English words dictionary
        english_words = set(nltk_words.words(paths=[NLTK_DATA_PATH]))
    except LookupError:
        st.warning("NLTK 'words' corpus not found. Cannot filter non-English words.")
        english_words = None # Flag that filtering cannot be done

    # --- Start Preprocessing ---
    text = text.lower()
    # Remove non-alphabetic characters (keeps spaces)
    text = re.sub("[^a-z\s]", "", text) # Keep spaces for splitting

    # Tokenize (simple split after regex cleaning)
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Filter for English words if the corpus was loaded
    if english_words is not None:
        words = [word for word in words if word in english_words]

    if not words:
        return [], "" # Return empty if no words remain after filtering

    # POS Tagging (Requires 'averaged_perceptron_tagger')
    try:
        tagged_words = pos_tag(words)
    except LookupError:
        st.error("NLTK 'averaged_perceptron_tagger' not found. Cannot perform lemmatization.")
        # Fallback: return words after stopword/English filtering but before lemmatization
        final_tokens = [w for w in words if len(w) >= DEFAULT_MIN_WORD_LENGTH]
        processed_text_string = " ".join(final_tokens)
        return final_tokens, processed_text_string
    except Exception as e:
        st.error(f"Error during POS tagging: {e}")
        return [], "" # Return empty on unexpected error


    # Lemmatization (Requires 'wordnet')
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatized_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tagged_words
        ]
    except LookupError:
        # This might happen if wordnet failed download despite earlier checks
        st.error("NLTK 'wordnet' resource not found during lemmatization.")
        # Fallback: return tagged words before lemmatization attempt
        final_tokens = [w for w in words if len(w) >= DEFAULT_MIN_WORD_LENGTH]
        processed_text_string = " ".join(final_tokens)
        return final_tokens, processed_text_string
    except Exception as e:
        st.error(f"Error during lemmatization: {e}")
        return [], "" # Return empty on unexpected error


    # Filter by minimum length
    final_tokens = [w for w in lemmatized_words if len(w) >= DEFAULT_MIN_WORD_LENGTH]

    # Prepare the joined string output
    processed_text_string = " ".join(final_tokens)

    return final_tokens, processed_text_string


@st.cache_data
def get_word_frequency_df(tokens: List[str], top_n: int = DEFAULT_TOP_N_WORDS) -> pd.DataFrame:
    """
    Calculates word frequencies and returns a DataFrame.

    Args:
        tokens: A list of word tokens.
        top_n: The number of most frequent words to return.

    Returns:
        A pandas DataFrame with 'Word' and 'Frequency' columns.
    """
    if not tokens:
        return pd.DataFrame(columns=['Word', 'Frequency'])
    fdist = FreqDist(tokens)
    common_words = fdist.most_common(top_n)
    df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    return df_freq

def plot_word_frequency(df_freq: pd.DataFrame):
    """Plots the word frequency bar chart using Matplotlib/Seaborn."""
    if df_freq.empty:
        st.warning("No words to plot after processing.")
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_freq) * 0.4))) # Dynamic height
    sns.barplot(x='Frequency', y='Word', data=df_freq, ax=ax, palette='viridis', hue='Word', dodge=False, legend=False)
    ax.set_title(f'Top {len(df_freq)} Most Frequent Words')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

@st.cache_data
def generate_word_cloud_image(processed_text_string: str) -> Optional[WordCloud]:
    """
    Generates a WordCloud object from processed text.

    Args:
        processed_text_string: A single string of space-separated processed words.

    Returns:
        A WordCloud object or None if generation fails or text is empty.
    """
    if not processed_text_string.strip():
        return None
    try:
        # Added max_words for performance/visuals on very large texts
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='viridis', max_words=200).generate(processed_text_string)
        return wordcloud
    except ValueError as e:
         # Handle cases where text might be too short or only contains ignored words
         print(f"WordCloud generation error: {e}") # Log error for debugging
         st.warning(f"Could not generate word cloud: {e}") # Show warning in UI
         return None

def display_word_cloud(wordcloud_obj: Optional[WordCloud]):
    """Displays the generated word cloud object using Matplotlib."""
    if wordcloud_obj:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_obj, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig) # Close the figure
    else:
        # Warning is now generated in generate_word_cloud_image if needed
        pass
        # st.warning("Could not generate word cloud. Input text might be too short or lack meaningful words after processing.")

@st.cache_data
def get_main_topics(processed_text_string: str, n_topics: int = DEFAULT_LDA_N_TOPICS, n_top_words: int = DEFAULT_LDA_N_TOP_WORDS) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Identifies main topics using Latent Dirichlet Allocation (LDA).

    Args:
        processed_text_string: A single string of space-separated processed words.
        n_topics: The number of topics to identify.
        n_top_words: The number of top words to display for each topic.

    Returns:
        A tuple containing:
            - A list of strings, each describing a topic, or None on failure.
            - An error message string if failed, otherwise None.
    """
    if not processed_text_string.strip():
         return None, "Input text is empty after processing."

    # Use CountVectorizer for LDA
    # Note: CountVectorizer has its own stop_words='english'. This might be redundant
    # with our preprocessing but is generally safe.
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        # Pass as a list/iterable
        tf = vectorizer.fit_transform([processed_text_string])
        feature_names = vectorizer.get_feature_names_out()

        # Check vocabulary and term count *after* fitting
        if not feature_names.any():
             return None, "Vectorization resulted in an empty vocabulary (min_df=2 might be too high, or text lacks diverse words after processing)."
        # Ensure enough features for the requested number of topics for LDA
        if tf.shape[1] < n_topics:
             return None, f"Insufficient unique terms ({tf.shape[1]}) after vectorization to form {n_topics} topic(s). Try reducing the number of topics."

    except ValueError as e:
        # This might catch errors if the input is completely empty or unprocessable by CountVectorizer
        return None, f"Text vectorization failed. Input might be unsuitable. Error: {e}"

    # Fit LDA model
    # Using n_init for stability, consider adjusting based on performance needs
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_init=10, learning_method='online') # 'online' can be faster
    try:
        lda.fit(tf)
    except ValueError as e:
         # Catch potential issues during LDA fitting (e.g., insufficient features for components)
         return None, f"LDA model fitting failed. Check if the number of topics ({n_topics}) is feasible for the processed text. Error: {e}"
    except Exception as e:
        # Catch other potential issues during LDA fitting
        return None, f"LDA model fitting failed. Error: {e}"


    topics_list = []
    if not hasattr(lda, 'components_') or lda.components_ is None:
         return None, "LDA model did not produce components."

    for topic_idx, topic_weights in enumerate(lda.components_):
        # Ensure we don't request more words than available features
        actual_top_words = min(n_top_words, len(feature_names))
        # Get indices of top words for this topic
        top_words_indices = topic_weights.argsort()[:-actual_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_list.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    if not topics_list:
        # This case might occur if LDA runs but fails to extract meaningful topics
        return None, "Could not extract meaningful topics from the LDA model components."

    return topics_list, None # Return list of topics and no error message

# --- Image Processing Functions ---
@st.cache_resource # Cache the model itself for the duration of the session
def load_vgg_model() -> Optional[Any]: # Using Any as TF types can be complex
    """Loads the VGG16 model pre-trained on ImageNet."""
    if not TF_AVAILABLE:
        st.error("TensorFlow is required for image analysis but not installed.")
        return None
    try:
        with st.spinner("Loading VGG16 model (this may take a moment)..."):
            model = VGG16(weights='imagenet')
        st.success("VGG16 model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading VGG16 model: {e}")
        return None

@st.cache_data # Cache results based on image bytes and model hash
def process_and_predict_image(image_bytes: bytes, _model_ref: Any, top_n: int = 5) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Processes an image and returns top N predictions using the VGG model.

    Args:
        image_bytes: The raw bytes of the image file.
        _model_ref: Reference to the loaded model (used for caching invalidation).
        top_n: The number of top predictions to return.

    Returns:
        A tuple containing:
            - A pandas DataFrame with 'label' and 'probability' columns, or None on failure.
            - An error message string if failed, otherwise None.
    """
    # Model is loaded outside and passed in, but check TF availability again
    if not TF_AVAILABLE:
        return None, "TensorFlow is not available."
    if _model_ref is None:
         return None, "VGG model is not loaded."

    model = _model_ref # Use the actual model object

    try:
        # Load image from bytes using PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure 3 channels

        # Resize and preprocess for VGG16
        img_resized = img.resize(VGG_IMG_SIZE, Image.Resampling.LANCZOS) # Use high-quality resampling
        img_array = img_to_array(img_resized) # Use the imported function
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded) # VGG16 specific preprocessing

        # Make prediction
        predictions = model.predict(img_preprocessed)
        # Decode predictions using ImageNet labels
        decoded_predictions = decode_predictions(predictions, top=top_n)[0] # Get top N results

        # Format results into a DataFrame
        results_list = [(label, f"{prob:.2%}") for _, label, prob in decoded_predictions]
        results_df = pd.DataFrame(results_list, columns=['Label', 'Probability'])

        return results_df, None # Return DataFrame and no error

    except Exception as e:
        # Catch potential errors during image loading, processing, or prediction
        st.error(f"Error processing or predicting image: {e}") # Show error in UI
        return None, f"Error processing or predicting image: {e}"


# --- Streamlit App Main Function ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Text & Image Analyzer")
    st.title("📝🖼️ Multi-Modal Analyzer")
    st.write("""
    Analyze text to find frequent words, generate word clouds, and identify topics (now with lemmatization!),
    or analyze images to predict their content using a pre-trained model.
    """)
    st.sidebar.header("Configuration")

    # --- NLTK Setup ---
    if not configure_nltk_path():
         st.stop() # Stop execution if NLTK path setup fails

    # Download necessary NLTK data (only if needed for text analysis)
    # These checks run once per session if successful due to caching/app state
    # Use resource paths for checking, download names for downloading
    nltk_resources_ok = []
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/stopwords', 'stopwords'))
    nltk_resources_ok.append(download_nltk_data_if_missing('tokenizers/punkt', 'punkt'))
    # Add downloads for the new preprocessing steps
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/wordnet', 'wordnet'))
    nltk_resources_ok.append(download_nltk_data_if_missing('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'))
    nltk_resources_ok.append(download_nltk_data_if_missing('corpora/words', 'words'))


    if not all(nltk_resources_ok):
         st.warning("One or more NLTK resources failed to download. Text analysis might be affected or incomplete.")
         # Depending on which failed, specific features might not work.
         # The preprocess_text function now has internal checks for some of these.


    # --- Input Selection ---
    st.sidebar.markdown("---")
    input_options = ['Text']
    if TF_AVAILABLE:
        input_options.append('Image')
    else:
        st.sidebar.warning("Image analysis disabled: TensorFlow not found.")

    input_type = st.sidebar.radio(
        "Select Input Type:",
        options=input_options,
        horizontal=True,
        key='input_type_selector'
    )

    # --- Analysis Parameters (Sidebar) ---
    st.sidebar.markdown("---")
    if input_type == 'Text':
        st.sidebar.subheader("Text Analysis Options")
        top_n_freq = st.sidebar.slider("Top N Frequent Words:", min_value=5, max_value=50, value=DEFAULT_TOP_N_WORDS, step=1)
        lda_n_topics = st.sidebar.number_input("Number of Topics (LDA):", min_value=1, max_value=10, value=DEFAULT_LDA_N_TOPICS, step=1)
        lda_n_words = st.sidebar.slider("Words per Topic (LDA):", min_value=3, max_value=20, value=DEFAULT_LDA_N_TOP_WORDS, step=1)
        # Note: min_word_length is now fixed in preprocess_text, could be made configurable here too

    elif input_type == 'Image':
        st.sidebar.subheader("Image Analysis Options")
        top_n_pred = st.sidebar.slider("Top N Predictions:", min_value=1, max_value=10, value=5, step=1)
        # Load the model once when Image analysis is selected
        vgg_model = load_vgg_model()


    # --- Main Panel for Input and Analysis ---
    st.markdown("---") # Separator in main panel

    analyze_button_pressed = False
    input_text = None
    uploaded_image = None

    if input_type == 'Text':
        input_text = st.text_area("Input Text:", height=250, placeholder="Paste or type your text here...")
        if st.button("Analyze Text", key="analyze_text_button"):
            if input_text and input_text.strip(): # Check if text is not just whitespace
                analyze_button_pressed = True
            else:
                st.warning("⚠️ Please enter some text to analyze.")

    elif input_type == 'Image' and TF_AVAILABLE:
        uploaded_image = st.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png'])
        # Disable button if no image is uploaded or model failed to load
        disable_image_button = (uploaded_image is None) or (vgg_model is None)
        if st.button("Analyze Image", key="analyze_image_button", disabled=disable_image_button):
             if uploaded_image is not None and vgg_model is not None:
                 analyze_button_pressed = True
             elif vgg_model is None:
                  st.error("Cannot analyze image: VGG model failed to load.")
             else:
                 st.warning("⚠️ Please upload an image file to analyze.")

    # --- Analysis Execution ---
    if analyze_button_pressed:
        st.markdown("---")
        st.subheader(f"🔎 {input_type} Analysis Results")

        if input_type == 'Text':
            with st.spinner("Processing text (including lemmatization)..."):
                # The new preprocess_text is called here
                tokens, processed_text_string = preprocess_text(input_text)

            if not tokens:
                st.warning("The input text resulted in no valid words after processing (check for only stopwords, non-English words, etc.).")
            else:
                # Use columns for better layout
                col1, col2 = st.columns(2)
                with col1:
                    # Word Frequency
                    st.markdown("##### 📊 Word Frequency Plot")
                    with st.spinner("Generating frequency plot..."):
                        df_freq = get_word_frequency_df(tokens, top_n=top_n_freq)
                        plot_word_frequency(df_freq)

                    # Topic Modeling
                    st.markdown("##### 💡 Main Topic(s) (LDA)")
                    with st.spinner("Identifying main topic(s)..."):
                         topics, error_msg = get_main_topics(processed_text_string, n_topics=lda_n_topics, n_top_words=lda_n_words)
                         if topics:
                             for topic in topics:
                                 st.write(topic)
                         else:
                             st.warning(f"Could not determine topic(s). Reason: {error_msg}")
                with col2:
                    # Word Cloud
                    st.markdown("##### ☁️ Word Cloud")
                    with st.spinner("Generating word cloud..."):
                        wordcloud_obj = generate_word_cloud_image(processed_text_string)
                        display_word_cloud(wordcloud_obj) # Handles None case internally

        elif input_type == 'Image' and uploaded_image is not None and vgg_model is not None:
            col_img, col_pred = st.columns([0.6, 0.4]) # Adjust column widths

            with col_img:
                st.markdown("##### Uploaded Image")
                image_bytes = uploaded_image.getvalue()
                try:
                    st.image(image_bytes, caption=f"Uploaded: {uploaded_image.name}", use_column_width='always')
                except Exception as e:
                    st.error(f"Could not display the uploaded image. Error: {e}")


            with col_pred:
                st.markdown(f"##### 🔍 Image Content Prediction (Top {top_n_pred})")
                with st.spinner("Analyzing image..."):
                    # Pass model reference for caching check
                    predictions_df, error_msg = process_and_predict_image(image_bytes, vgg_model, top_n=top_n_pred)

                    if predictions_df is not None:
                        # Use st.dataframe for better table display
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                    else:
                        # Error message already shown in process_and_predict_image
                        st.error(f"Image analysis failed. {error_msg or ''}")

    st.markdown("---")
    st.caption("Powered by Streamlit, NLTK, Scikit-learn, Matplotlib, Seaborn, WordCloud" + (", TensorFlow/Keras" if TF_AVAILABLE else ""))

if __name__ == "__main__":
    # Suggest creating a requirements file if it doesn't exist
    # print("Consider creating a requirements.txt file for dependencies:")
    # print("pip freeze > requirements.txt")
    main()
