import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2 # Pour le redimensionnement d'image et la lecture
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis.lda_model # Pour charger la visualisation
import pyLDAvis # Pour afficher
from sklearn.manifold import TSNE # Pour recalculer t-SNE avec le nouveau point
from sklearn.base import BaseEstimator, TransformerMixin # Nécessaire pour les classes personnalisées
from nltk.corpus import stopwords, wordnet, words # Pour TextPreprocessor
from nltk.stem import WordNetLemmatizer # Pour TextPreprocessor
from nltk import pos_tag # Pour TextPreprocessor
import re # Pour TextPreprocessor
from sklearn.feature_extraction.text import CountVectorizer # Pour TopicModeler

# --- Configuration et Constantes ---
OUTPUT_DIR = "model_outputs"
EFFICIENTNET_INPUT_SIZE = (224, 224) # Doit correspondre à l'entraînement

# TensorFlow/Keras (uniquement pour le pré-traitement EfficientNet et le chargement du modèle si non sauvegardé)
# Si feature_extractor est sauvegardé via joblib (non recommandé pour les modèles Keras),
# il faudrait le charger différemment. Ici, on suppose qu'on le recrée.
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as tf_preprocess_input

# --- Définitions des Classes Personnalisées (identiques à celles du script d'entraînement) ---
english_words_global = set(words.words()) # Assurez-vous que nltk.download('words') a été exécuté
GLOBAL_STOP_WORDS = set(stopwords.words("english"))
GLOBAL_LEMMATIZER = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, english_words_set=english_words_global, min_len_word=3, rejoin=True):
        self.english_words_set = english_words_set
        self.min_len_word = min_len_word
        self.rejoin = rejoin

    def fit(self, X, y=None):
        return self

    def _preprocess_single_text(self, text_content): # Renommé pour éviter conflit avec transform
        if not isinstance(text_content, str):
            text_content = str(text_content)
        text_content = text_content.lower()
        text_content = re.sub("[^a-z]", " ", text_content)
        tokens = text_content.split()
        processed_tokens = []
        for word, pos_tag_val in pos_tag(tokens):
            if word not in GLOBAL_STOP_WORDS and \
               (not self.english_words_set or word in self.english_words_set):
                lemma = GLOBAL_LEMMATIZER.lemmatize(word, get_wordnet_pos(pos_tag_val))
                if len(lemma) >= self.min_len_word:
                    processed_tokens.append(lemma)
        if self.rejoin:
            return " ".join(processed_tokens)
        return processed_tokens

    def transform(self, X_texts): # Renommé pour éviter conflit avec _preprocess_single_text
        return [self._preprocess_single_text(doc) for doc in X_texts]

class TopicModeler(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics_model=4):
        self.num_topics_model = num_topics_model
        # Ces attributs seront chargés par joblib s'ils ont été sauvegardés avec l'instance
        # ou réinitialisés/ré-entrainés si ce n'est pas le cas.
        # Pour le chargement, il est crucial que ces objets soient sérialisables par joblib.
        self.count_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=1000, stop_words=None)
        self.lda_model = None # Sera chargé ou créé
        self.topic_vectors = None
        self.dtm = None

    def fit(self, X, y=None):
        # Normalement, le pipeline est déjà "fit" lors de la sauvegarde.
        # Cette méthode est ici pour la compatibilité avec l'interface sklearn.
        # Si vous chargez un pipeline pré-entraîné, vous n'appellerez pas fit() à nouveau.
        raise NotImplementedError("This TopicModeler is intended to be loaded as part of a pre-trained pipeline. Fit should not be called directly in the Streamlit app.")

    def transform(self, X):
        # Le lda_model et count_vectorizer sont supposés être chargés par joblib
        # comme faisant partie de l'état de l'instance TopicModeler sauvegardée.
        if self.lda_model is None or not hasattr(self.lda_model, 'components_'):
             st.warning("Le modèle LDA dans TopicModeler n'est pas correctement chargé/configuré.")
             return np.array([])
        dtm_new = self.count_vectorizer.transform(X) # count_vectorizer doit être fit
        return self.lda_model.transform(dtm_new)

# --- Fonctions de Chargement des Artefacts ---
@st.cache_resource # Cache les ressources lourdes
def load_all_artifacts():
    artifacts = {}
    required_text_files = {
        "text_pipeline": "text_processing_pipeline.joblib",
        "text_lda_model_main": "text_sklearn_lda_model.joblib", # Le modèle LDA principal
        "text_cv_main": "text_sklearn_count_vectorizer.joblib", # Le CV principal
        "text_dtm_main": "text_sklearn_dtm.joblib", # DTM des données originales
        "text_topic_vectors_main": "text_sklearn_topic_vectors.joblib", # Topic vectors des données originales
        "text_dominant_topics_main": "text_sklearn_dominant_topics.joblib", # Dominant topics des données originales
        "text_num_topics": "text_sklearn_lda_model.joblib", # On va extraire num_topics de ce modèle
        "text_pyldavis_html_path": "text_lda_visualization.html"
    }
    required_image_files = {
        "image_scaler": "image_scaler.joblib",
        "image_pca": "image_pca.joblib",
        "image_model": "image_random_forest_model.joblib",
        "image_label_encoder": "image_label_encoder.joblib",
        "image_features_original": "image_features_X_np.joblib", # Features des images originales
        "image_labels_original_str": "image_labels_y_np_original.joblib" # Labels string des images originales
    }

    # Chargement des artefacts textuels
    for key, filename in required_text_files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            if filename.endswith(".html"):
                artifacts[key] = path # Store path for HTML
            else:
                artifacts[key] = joblib.load(path)
        else:
            st.error(f"Artefact textuel manquant : {filename} dans {OUTPUT_DIR}")
            return None
    # Extraire num_topics du modèle LDA principal
    if artifacts.get("text_lda_model_main"):
        artifacts["text_num_topics_actual"] = artifacts["text_lda_model_main"].n_components
    else: # Fallback si le modèle principal n'est pas là (devrait l'être)
        artifacts["text_num_topics_actual"] = artifacts.get("text_pipeline").named_steps["topic_modeling"].num_topics_model


    # Chargement des artefacts images
    for key, filename in required_image_files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            # Label encoder est optionnel si les labels étaient déjà numériques
            if key == "image_label_encoder":
                artifacts[key] = None
                st.warning(f"Artefact image optionnel manquant : {filename}. Les prédictions seront numériques.")
            else:
                st.error(f"Artefact image manquant : {filename} dans {OUTPUT_DIR}")
                return None

    # Recréer le feature_extractor pour les images (EfficientNetB0)
    # Les poids sont chargés depuis 'imagenet', donc pas besoin de le sauvegarder/charger avec joblib
    try:
        artifacts["image_feature_extractor"] = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    except Exception as e:
        st.error(f"Erreur lors de la création de EfficientNetB0 : {e}")
        return None

    st.success("Tous les artefacts ont été chargés avec succès !")
    return artifacts

# --- Fonctions Utilitaires pour l'Affichage ---
def display_text_wordcloud(topic_idx, lda_model, vectorizer, num_top_words=20):
    """Affiche le WordCloud pour un topic donné."""
    if lda_model is None or vectorizer is None:
        st.warning("Modèle LDA ou Vectorizer non disponible pour le WordCloud.")
        return

    feature_names = vectorizer.get_feature_names_out()
    topic_word_dist = lda_model.components_[topic_idx]
    
    # S'assurer que les indices sont valides
    valid_indices = [i for i in topic_word_dist.argsort()[:-num_top_words-1:-1] if i < len(feature_names)]
    if not valid_indices:
        st.write("Aucun mot trouvé pour ce topic.")
        return

    wc_dict = {feature_names[i]: topic_word_dist[i] for i in valid_indices}
    
    if not wc_dict:
        st.write("Aucun mot trouvé pour ce topic après filtrage.")
        return

    wc = WordCloud(background_color="white", width=600, height=300)
    wc.generate_from_frequencies(wc_dict)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def plot_tsne_with_new_point(original_vectors, new_vector, original_labels, new_label, title, point_type="Texte"):
    """Affiche le t-SNE avec le nouveau point mis en évidence."""
    if original_vectors is None or new_vector is None:
        st.warning(f"Données manquantes pour le t-SNE {point_type}.")
        return

    all_vectors = np.vstack([original_vectors, new_vector.reshape(1, -1)])
    all_labels_str = [str(l) for l in original_labels] + [f"Nouveau {point_type} ({str(new_label)})"]

    perplexity_val = min(30, all_vectors.shape[0] - 1) if all_vectors.shape[0] > 1 else 5
    if perplexity_val <=0: perplexity_val = 1 # Ensure perplexity is positive

    tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, init='pca', n_iter=300) # n_iter réduit pour rapidité
    
    try:
        tsne_results = tsne_model.fit_transform(all_vectors)
    except ValueError as e:
        st.error(f"Erreur lors du calcul du t-SNE ({point_type}): {e}. Perplexité: {perplexity_val}, Nb points: {all_vectors.shape[0]}")
        st.write("Cela peut arriver si le nombre de points est trop faible ou si la perplexité est mal ajustée.")
        return


    df_tsne = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df_tsne['label'] = all_labels_str
    df_tsne['size'] = [5] * len(original_vectors) + [15] # Mettre en évidence le nouveau point
    df_tsne['symbol'] = ['circle'] * len(original_vectors) + ['star']

    fig = px.scatter(df_tsne, x='x', y='y', color='label', title=title,
                     size='size', symbol='symbol',
                     hover_data={'label': True, 'size': False, 'symbol': False})
    fig.update_layout(legend_title_text=f"Topic/Catégorie ({point_type})")
    st.plotly_chart(fig, use_container_width=True)

# --- Fonctions de Prédiction ---
def predict_text_topic(raw_text, artifacts):
    """Prédit le topic pour un texte donné en utilisant le pipeline et le modèle LDA principal."""
    text_pipeline = artifacts.get("text_pipeline")
    lda_main = artifacts.get("text_lda_model_main")
    cv_main = artifacts.get("text_cv_main")

    if not text_pipeline or not lda_main or not cv_main:
        st.error("Pipeline de texte ou modèle LDA/CV principal non chargé.")
        return None, None, None

    # 1. Prétraitement via le pipeline (juste l'étape de preprocessing)
    preprocessor = text_pipeline.named_steps['preprocessing']
    processed_text_list = preprocessor.transform([raw_text]) # Doit retourner une liste de strings
    
    if not processed_text_list or not processed_text_list[0].strip():
        st.warning("Le texte est vide après prétraitement.")
        return None, None, None
    
    processed_text_str = processed_text_list[0]

    # 2. Vectorisation avec le CountVectorizer principal
    new_text_dtm = cv_main.transform([processed_text_str])

    # 3. Prédiction de la distribution des topics avec le modèle LDA principal
    new_text_topic_distribution = lda_main.transform(new_text_dtm)
    
    if new_text_topic_distribution.shape[0] == 0:
        st.error("La prédiction de topic a échoué (distribution vide).")
        return None, None, None

    dominant_topic_idx = np.argmax(new_text_topic_distribution[0])
    
    # Obtenir les mots-clés pour l'affichage
    feature_names = cv_main.get_feature_names_out()
    topic_keywords_list = []
    for i in range(lda_main.n_components):
        top_words_indices = lda_main.components_[i].argsort()[:-6:-1] # Top 5 mots
        topic_keywords_list.append([feature_names[j] for j in top_words_indices])
    
    synthetic_label = f"Topic {dominant_topic_idx}: " + ", ".join(topic_keywords_list[dominant_topic_idx])
    
    return dominant_topic_idx, synthetic_label, new_text_topic_distribution[0]


def predict_image_category(image_pil, artifacts):
    """Prédit la catégorie pour une image donnée."""
    scaler = artifacts.get("image_scaler")
    pca = artifacts.get("image_pca")
    model = artifacts.get("image_model")
    label_encoder = artifacts.get("image_label_encoder")
    feature_extractor = artifacts.get("image_feature_extractor")

    if not all([scaler, pca, model, feature_extractor]):
        st.error("Un ou plusieurs artefacts pour la prédiction d'image sont manquants.")
        return None, None

    try:
        # Prétraitement de l'image
        img_array = np.array(image_pil.convert("RGB")) # Assurer RGB
        img_resized = cv2.resize(img_array, EFFICIENTNET_INPUT_SIZE)
        img_expanded = np.expand_dims(img_resized, axis=0)
        img_preprocessed = tf_preprocess_input(img_expanded)

        # Extraction des features
        features_new = feature_extractor.predict(img_preprocessed, verbose=0)[0]
        features_new_reshaped = features_new.reshape(1, -1)

        # Scaling et PCA
        features_new_std = scaler.transform(features_new_reshaped)
        features_new_pca = pca.transform(features_new_std)

        # Prédiction
        prediction_numeric = model.predict(features_new_pca)[0]
        proba = model.predict_proba(features_new_pca)[0] # Pourrait être utile

        predicted_label_str = str(prediction_numeric)
        if label_encoder and hasattr(label_encoder, 'classes_') and prediction_numeric < len(label_encoder.classes_):
            predicted_label_str = label_encoder.inverse_transform([prediction_numeric])[0]
        
        return predicted_label_str, features_new # Retourner les features brutes pour t-SNE
    except Exception as e:
        st.error(f"Erreur lors de la prédiction d'image : {e}")
        return None, None

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("🔮 Analyseur de Topics et Catégories 🔮")

artifacts = load_all_artifacts()

if artifacts:
    choice = st.sidebar.radio("Que souhaitez-vous analyser ?", ("Texte 📜", "Image 🖼️"))

    if choice == "Texte 📜":
        st.header("Analyse de Texte")
        user_text = st.text_area("Entrez votre texte ici :", height=150, placeholder="Ex: J'adore ce nouveau t-shirt bleu, il est parfait pour l'été et très confortable.")

        if st.button("Analyser le Texte", key="text_analysis_button"):
            if user_text.strip():
                predicted_topic, synthetic_label, new_text_topic_vec = predict_text_topic(user_text, artifacts)

                if predicted_topic is not None:
                    st.subheader(f"📝 Topic Prédit : {synthetic_label}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### ☁️ WordCloud du Topic")
                        display_text_wordcloud(predicted_topic, 
                                               artifacts["text_lda_model_main"], 
                                               artifacts["text_cv_main"])
                    with col2:
                        st.markdown("#### 🗺️ Visualisation t-SNE des Documents")
                        if artifacts.get("text_topic_vectors_main") is not None and \
                           artifacts.get("text_dominant_topics_main") is not None and \
                           new_text_topic_vec is not None:
                            plot_tsne_with_new_point(artifacts["text_topic_vectors_main"],
                                                     new_text_topic_vec,
                                                     artifacts["text_dominant_topics_main"],
                                                     predicted_topic,
                                                     "Projection t-SNE des Topics Textuels",
                                                     point_type="Texte")
                        else:
                            st.warning("Données t-SNE pour le texte non disponibles.")
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Visualisation Interactive pyLDAvis")
                    pyldavis_html_path = artifacts.get("text_pyldavis_html_path")
                    if pyldavis_html_path and os.path.exists(pyldavis_html_path):
                        try:
                            with open(pyldavis_html_path, 'r', encoding='utf-8') as f:
                                html_string = f.read()
                            st.components.v1.html(html_string, width=None, height=800, scrolling=True)
                            # Alternative: Charger les données et préparer à la volée (plus lourd)
                            # vis_data = pyLDAvis.lda_model.prepare(
                            #    artifacts["text_lda_model_main"],
                            #    artifacts["text_dtm_main"],
                            #    artifacts["text_cv_main"],
                            #    mds="mmds" # ou 'tsne'
                            # )
                            # pyLDAvis.display(vis_data) # Ne fonctionne pas directement dans st, utiliser st.pyLDAvis_html
                        except Exception as e_pylda_disp:
                            st.error(f"Erreur lors de l'affichage de pyLDAvis : {e_pylda_disp}")
                            st.markdown(f"Vous pouvez ouvrir la visualisation [ici]({pyldavis_html_path}). (Nécessite de servir le fichier localement si Streamlit est distant)")
                    else:
                        st.warning("Fichier de visualisation pyLDAvis non trouvé.")

                else:
                    st.error("Impossible de prédire le topic pour le texte fourni.")
            else:
                st.warning("Veuillez entrer du texte.")

    elif choice == "Image 🖼️":
        st.header("Analyse d'Image")
        uploaded_file = st.file_uploader("Chargez une image :", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Image Chargée", use_container_width=True)

            if st.button("Analyser l'Image", key="image_analysis_button"):
                with col_img2:
                    with st.spinner("Analyse de l'image en cours..."):
                        predicted_category, new_image_features_raw = predict_image_category(image, artifacts)
                    
                    if predicted_category is not None:
                        st.subheader(f"🏷️ Catégorie Prédite : {predicted_category}")
                        
                        st.markdown("#### 🗺️ Visualisation t-SNE des Images")
                        if artifacts.get("image_features_original") is not None and \
                           artifacts.get("image_labels_original_str") is not None and \
                           new_image_features_raw is not None:
                            
                            # Pour le t-SNE, nous avons besoin des features scalées et réduites par PCA
                            # comme pour l'entraînement, mais ici on le fait juste pour le nouveau point.
                            # Les points originaux sont déjà des features brutes.
                            # Donc, on va projeter TOUTES les features brutes (originales + nouvelle)
                            
                            plot_tsne_with_new_point(artifacts["image_features_original"],
                                                     new_image_features_raw, # Utiliser les features brutes extraites
                                                     artifacts["image_labels_original_str"],
                                                     predicted_category,
                                                     "Projection t-SNE des Catégories d'Images",
                                                     point_type="Image")
                        else:
                            st.warning("Données t-SNE pour les images non disponibles.")
                    else:
                        st.error("Impossible de prédire la catégorie pour l'image fournie.")
else:
    st.error("Erreur critique : Impossible de charger les artefacts nécessaires. Vérifiez que le script principal a bien été exécuté et que le dossier 'model_outputs' est correct.")

st.sidebar.markdown("---")
st.sidebar.info("Application développée avec Streamlit pour la classification de textes et d'images.")
