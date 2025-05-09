import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import os
# import logging # logging n'était pas utilisé activement dans l'UI Streamlit, peut être réactivé si besoin pour le débogage serveur.
from collections import Counter # Présent dans les imports originaux, conservé.

# Importations NLTK (avec gestion des téléchargements)
import nltk
from nltk.corpus import stopwords, wordnet, words
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Importations Scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE # Utilisé pour LDA et potentiellement pour l'image si on le génère ici

# Imports TensorFlow/Keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib # Pour charger les objets scikit-learn sauvegardés

# Importations Visualisation
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objects as go # Non utilisé activement dans la version fusionnée, peut être rajouté si besoin.
from wordcloud import WordCloud

# --- Configuration Générale ---
# Chemins pour l'analyse d'image
SCALER_PATH = 'scaler.joblib'
PCA_PATH = 'pca.joblib'
MODEL_PATH = 'random_forest_model.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'
TSNE_PLOT_IMAGE_PATH = 'tsne_visualization.png'
EFFICIENTNET_INPUT_SIZE = (224, 224)

# Chemin pour les données d'analyse de texte (LDA)
LDA_DATA_FILEPATH = "negative_reviews5000.csv"

# --- Configuration et Téléchargements NLTK (Mise en cache) ---
@st.cache_resource
def download_nltk_resources():
    """Télécharge les ressources NLTK nécessaires."""
    resources_config = {
        "stopwords": "corpora/stopwords",
        "words": "corpora/words",
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger"
    }
    all_successful = True
    for resource_name, resource_path in resources_config.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            st.info(f"Téléchargement de la ressource NLTK : {resource_name}")
            try:
                nltk.download(resource_name, quiet=True)
                nltk.data.find(resource_path) # Vérifier à nouveau
                st.success(f"Ressource NLTK '{resource_name}' téléchargée.")
            except Exception as e:
                st.error(f"Échec du téléchargement de '{resource_name}': {e}")
                all_successful = False
    return all_successful

# Initialisation des variables globales NLTK (après téléchargement)
NLTK_RESOURCES_INITIALIZED = False
stop_words_set = set()
english_words_set = set()

def initialize_nltk_globals():
    global NLTK_RESOURCES_INITIALIZED, stop_words_set, english_words_set
    if not NLTK_RESOURCES_INITIALIZED:
        if download_nltk_resources():
            try:
                stop_words_set = set(stopwords.words("english"))
                english_words_set = set(words.words())
                NLTK_RESOURCES_INITIALIZED = True
            except LookupError as e:
                st.error(f"Erreur lors de l'initialisation des ressources NLTK après téléchargement: {e}")
                NLTK_RESOURCES_INITIALIZED = False
        else:
            st.error("Impossible de charger les ressources NLTK pour l'analyse de texte. Certaines fonctionnalités pourraient ne pas fonctionner.")
            NLTK_RESOURCES_INITIALIZED = False

# --- Fonctions de Traitement de Texte (LDA) ---

def get_wordnet_pos(treebank_tag):
    """Convertit les étiquettes POS de Treebank en format WordNet."""
    if treebank_tag.startswith("J"): return wordnet.ADJ
    elif treebank_tag.startswith("V"): return wordnet.VERB
    elif treebank_tag.startswith("N"): return wordnet.NOUN
    elif treebank_tag.startswith("R"): return wordnet.ADV
    else: return wordnet.NOUN # Par défaut

def preprocess_text_lda(text, min_len_word=3, rejoin=False):
    """Pré-traite un texte pour LDA (minuscules, regex, stopwords, mots anglais, lemmatisation)."""
    global stop_words_set, english_words_set, NLTK_RESOURCES_INITIALIZED
    if not NLTK_RESOURCES_INITIALIZED: # S'assurer que les globals sont prêts
        initialize_nltk_globals()
        if not NLTK_RESOURCES_INITIALIZED: # Si toujours pas prêt après tentative
            st.warning("Preprocessing de texte limité car les ressources NLTK ne sont pas chargées.")
            # Fallback simple si NLTK n'est pas prêt
            if not isinstance(text, str): text = str(text)
            text_simple_clean = re.sub("[^a-z\s]", "", text.lower())
            return " ".join(text_simple_clean.split()) if rejoin else text_simple_clean.split()

    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub("[^a-z\s]", "", text)
    
    words_list = [word for word in text.split() if word not in stop_words_set]

    if english_words_set:
        filtered_corpus = [word for word in words_list if word in english_words_set]
    else:
        st.warning("Le filtrage par mots anglais est désactivé (dictionnaire non chargé).")
        filtered_corpus = words_list

    try:
        lemmatizer = WordNetLemmatizer()
        tagged_words = pos_tag(filtered_corpus)
        lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words]
        final_tokens = [w for w in lemmatized_tokens if len(w) >= min_len_word]
    except Exception as e:
        st.error(f"Erreur NLTK pendant le preprocessing (POS tag/lemmatisation): {e}")
        final_tokens = [w for w in filtered_corpus if len(w) >= min_len_word] # Fallback

    return " ".join(final_tokens) if rejoin else final_tokens

class TextPreprocessorLDA(BaseEstimator, TransformerMixin):
    """Transformer Scikit-learn pour le prétraitement de texte LDA."""
    def __init__(self, min_len_word=3, rejoin=True):
        self.min_len_word = min_len_word
        self.rejoin = rejoin

    def fit(self, X, y=None): return self
    def transform(self, X):
        return [preprocess_text_lda(str(doc), min_len_word=self.min_len_word, rejoin=self.rejoin) for doc in X]

class TopicModelerLDA(BaseEstimator, TransformerMixin):
    """Transformer Scikit-learn pour la modélisation de sujets LDA."""
    def __init__(self, num_topics=4, max_features=1000, random_state=42):
        self.num_topics = num_topics
        self.max_features = max_features
        self.random_state = random_state
        self.lda_model = None
        self.count_vectorizer = None
        self.topic_vectors = None # Vecteurs pour les données d'entraînement

    def fit(self, X, y=None):
        st.info(f"Entraînement du modèle LDA avec {self.num_topics} topics...")
        self.count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=self.max_features, stop_words="english")
        dtm = self.count_vectorizer.fit_transform(X)
        
        if dtm.shape[0] == 0 or dtm.shape[1] == 0:
            st.warning("Matrice DTM vide. LDA non entraîné.")
            self.lda_model = None
            return self
        
        actual_num_topics = min(self.num_topics, dtm.shape[1])
        if actual_num_topics < 1:
            st.error("Aucune feature trouvée après vectorisation. LDA non entraîné.")
            self.lda_model = None
            return self
        if actual_num_topics < self.num_topics:
            st.warning(f"Nombre de features ({dtm.shape[1]}) insuffisant pour {self.num_topics} topics. Réduction du nombre de topics à {actual_num_topics}.")
            
        self.lda_model = LatentDirichletAllocation(
            n_components=actual_num_topics, 
            learning_method="online", 
            random_state=self.random_state, 
            max_iter=10, 
            n_jobs=-1
        )
        try:
            self.lda_model.fit(dtm)
            self.topic_vectors = self.lda_model.transform(dtm)
            st.success("Entraînement LDA terminé.")
        except Exception as e:
            st.error(f"Erreur pendant l'entraînement du modèle LDA : {e}")
            self.lda_model = None
        return self

    def transform(self, X):
        if not self.lda_model or not self.count_vectorizer:
            st.error("Modèle LDA ou Vectorizer non entraîné.")
            num_output_topics = self.num_topics if self.lda_model is None else self.lda_model.n_components
            return np.zeros((len(X), num_output_topics))
        try:
            dtm_new = self.count_vectorizer.transform(X)
            return self.lda_model.transform(dtm_new)
        except Exception as e:
            st.error(f"Erreur pendant la transformation LDA : {e}")
            return np.zeros((len(X), self.lda_model.n_components))

@st.cache_data
def load_lda_data(filepath=LDA_DATA_FILEPATH):
    """Charge les données pour LDA depuis un fichier CSV."""
    if not os.path.exists(filepath):
        st.error(f"Fichier de données LDA non trouvé : {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if 'text' not in df.columns:
            st.error("Colonne 'text' manquante dans le fichier de données LDA.")
            return None
        df = df.dropna(subset=['text'])
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Erreur de chargement du fichier de données LDA : {e}")
        return None

@st.cache_resource
def get_trained_lda_pipeline(data, num_topics=4):
    """Crée et entraîne le pipeline de traitement de texte pour LDA."""
    if data is None or data.empty: 
        st.error("Données invalides ou vides pour l'entraînement du pipeline LDA.")
        return None
    
    global NLTK_RESOURCES_INITIALIZED # Vérifier si NLTK est prêt
    if not NLTK_RESOURCES_INITIALIZED:
        initialize_nltk_globals()
        if not NLTK_RESOURCES_INITIALIZED:
            st.error("Pipeline LDA ne peut être entraîné sans les ressources NLTK.")
            return None
            
    pipeline = Pipeline([
        ("preprocessing", TextPreprocessorLDA(min_len_word=3, rejoin=True)),
        ("topic_modeling", TopicModelerLDA(num_topics=num_topics, random_state=42))
    ])
    with st.spinner(f"Entraînement du pipeline LDA avec {num_topics} topics (cela peut prendre un moment)..."):
        try:
            pipeline.fit(data["text"])
            if pipeline.named_steps['topic_modeling'].lda_model is None:
                st.error("Échec de l'entraînement du modèle LDA au sein du pipeline.")
                return None
            st.success("Pipeline LDA entraîné.")
        except Exception as e:
            st.error(f"Erreur majeure lors de l'entraînement du pipeline LDA: {e}")
            return None
    return pipeline

# --- Fonctions de Visualisation pour LDA ---

def plot_wordcloud_lda(pipeline, predicted_topic_idx=None):
    """Affiche les nuages de mots pour chaque sujet LDA."""
    topic_modeler = pipeline.named_steps.get('topic_modeling')
    if not topic_modeler or not topic_modeler.lda_model or not topic_modeler.count_vectorizer or \
       not hasattr(topic_modeler.count_vectorizer, 'vocabulary_') or not topic_modeler.count_vectorizer.vocabulary_:
        st.warning("Word Clouds non disponibles (modèle/vectorizer LDA non prêt).")
        return
    
    feature_names = topic_modeler.count_vectorizer.get_feature_names_out()
    actual_num_topics = topic_modeler.lda_model.n_components
    
    ncols = 2
    nrows = int(np.ceil(actual_num_topics / ncols))
    if nrows == 0: return

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 5), squeeze=False)
    axes = axes.flatten()
    for i in range(actual_num_topics):
        ax = axes[i]
        try:
            topic_words_dist = topic_modeler.lda_model.components_[i]
            top_word_indices = topic_words_dist.argsort()[:-min(21, len(feature_names)+1):-1]
            wc_dict = {feature_names[idx]: topic_words_dist[idx] for idx in top_word_indices if idx < len(feature_names)}
            
            if not wc_dict:
                ax.set_title(f"Topic {i} (Vide)"); ax.axis("off"); continue
            
            wc = WordCloud(background_color="white", width=400, height=200, max_words=50).generate_from_frequencies(wc_dict)
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            title = f"Topic {i}"
            if predicted_topic_idx is not None and predicted_topic_idx == i: 
                title += " (Votre Texte)"
                ax.set_title(title, color='red', fontweight='bold')
                rect = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='red', lw=3, transform=ax.transAxes, clip_on=False)
                ax.add_patch(rect)
            else:
                ax.set_title(title)
        except Exception as e:
            st.error(f"Erreur génération WordCloud pour Topic {i}: {e}")
            ax.set_title(f"Topic {i} (Erreur)"); ax.axis("off")


    for j in range(actual_num_topics, nrows * ncols): 
        if j < len(axes): axes[j].axis("off") # Masquer les axes inutilisés
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle("Word Clouds par Sujet LDA", fontsize=16)
    st.pyplot(fig); plt.close(fig)

@st.cache_data
def get_tsne_lda_results(_pipeline):
    """Calcule t-SNE sur les vecteurs de sujet des données d'entraînement LDA."""
    topic_modeler = _pipeline.named_steps.get('topic_modeling')
    if not topic_modeler or not topic_modeler.lda_model or topic_modeler.topic_vectors is None:
        st.warning("Vecteurs de sujet d'entraînement non disponibles pour t-SNE LDA.")
        return None, None
    
    original_topic_vectors = topic_modeler.topic_vectors
    if original_topic_vectors.shape[0] <= 1:
        st.warning("t-SNE LDA nécessite au moins 2 échantillons d'entraînement.")
        return None, None

    with st.spinner("Calcul de la réduction t-SNE pour les données LDA (cela peut prendre du temps)..."):
        perplexity_value = min(30, original_topic_vectors.shape[0] - 1)
        if perplexity_value <= 0 : perplexity_value = max(1, original_topic_vectors.shape[0] -1) # Ensure perplexity > 0
        
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='pca', learning_rate='auto', n_iter=300)
        try:
            tsne_vectors_lda = tsne_model.fit_transform(original_topic_vectors)
            dominant_topics_lda = np.argmax(original_topic_vectors, axis=1)
            st.success("Calcul t-SNE pour LDA terminé.")
            return tsne_vectors_lda, dominant_topics_lda
        except Exception as e:
            st.error(f"Erreur lors du calcul t-SNE pour LDA: {e}")
            return None, None


def plot_tsne_lda_streamlit(tsne_vectors, dominant_topics, num_actual_topics):
    """Affiche la visualisation t-SNE pour LDA avec Plotly."""
    if tsne_vectors is None or dominant_topics is None:
        st.warning("Données t-SNE LDA manquantes pour la visualisation.")
        return
    if len(tsne_vectors) != len(dominant_topics):
        st.error("Incohérence de taille entre vecteurs t-SNE et topics dominants pour LDA.")
        return

    df_lda_tsne = pd.DataFrame(tsne_vectors, columns=['x', 'y'])
    df_lda_tsne['dominant_topic'] = dominant_topics.astype(str)
    
    colors = px.colors.qualitative.Plotly
    color_map = {str(i): colors[i % len(colors)] for i in range(num_actual_topics)}
    
    fig = px.scatter(df_lda_tsne, x='x', y='y', color='dominant_topic',
                     color_discrete_map=color_map,
                     title='Visualisation t-SNE des Sujets LDA (Données d\'entraînement)',
                     labels={'dominant_topic': 'Sujet Dominant'},
                     template='plotly_white', opacity=0.7)
    fig.update_layout(legend_title_text='Topics LDA')
    st.plotly_chart(fig, use_container_width=True)

# --- Fonctions pour l'Analyse d'Image ---
@st.cache_resource
def load_feature_extractor_image():
    """Charge le modèle EfficientNetB0 pour l'extraction de caractéristiques images."""
    model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    return model

feature_extractor_image_model = load_feature_extractor_image()

@st.cache_resource
def load_sklearn_object_image(path):
    """Charge un objet scikit-learn sauvegardé pour l'analyse d'image."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'objet image depuis {path}: {e}")
            return None
    else:
        if path != LABEL_ENCODER_PATH: # Avertir seulement si ce n'est pas l'encodeur optionnel
             st.warning(f"Avertissement : Fichier image non trouvé à {path}. La prédiction d'image pourrait ne pas fonctionner.")
        return None

scaler_img_analysis = load_sklearn_object_image(SCALER_PATH)
pca_img_analysis = load_sklearn_object_image(PCA_PATH)
classifier_img_analysis = load_sklearn_object_image(MODEL_PATH)
label_encoder_img_analysis = load_sklearn_object_image(LABEL_ENCODER_PATH)

def predict_image_topic(image_array_bgr, feature_extractor_model, scaler_obj, pca_obj, classifier_model, label_encoder_obj=None):
    """Prédit le topic d'une image."""
    if image_array_bgr is None: return "Erreur: Image non valide."
    try:
        img_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, EFFICIENTNET_INPUT_SIZE)
        img_expanded = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_expanded) # Spécifique à EfficientNet
        
        features_new_image = feature_extractor_model.predict(img_preprocessed, verbose=0)[0]
        features_new_image_reshaped = features_new_image.reshape(1, -1)
        
        if scaler_obj is None: return "Erreur: Scaler pour image non chargé."
        if pca_obj is None: return "Erreur: PCA pour image non chargé."
        if classifier_model is None: return "Erreur: Modèle de classification image non chargé."
        
        features_new_image_std = scaler_obj.transform(features_new_image_reshaped)
        features_new_image_pca = pca_obj.transform(features_new_image_std)
        prediction_numeric = classifier_model.predict(features_new_image_pca)[0]
        
        if label_encoder_obj and hasattr(label_encoder_obj, 'classes_'):
            return label_encoder_obj.inverse_transform([prediction_numeric])[0]
        else:
            return str(prediction_numeric) # Retourne la classe numérique si pas d'encodeur
    except Exception as e:
        return f"Erreur lors de la prédiction d'image : {e}"

# --- Interface Utilisateur Streamlit ---
st.set_page_config(page_title="Analyse Multi-Modale", layout="wide")
st.title("🔬 Analyse Multi-Modale : Texte (LDA) & Image (Classification)")

# Initialiser NLTK pour toute l'application (surtout pour la partie LDA)
initialize_nltk_globals()


analysis_type = st.sidebar.radio("Choisissez le type d'analyse :", ("Analyse de Texte (LDA)", "Analyse d'Image (Classification)"))
st.sidebar.markdown("---")

if analysis_type == "Analyse de Texte (LDA)":
    st.header("📝 Analyse de Texte (LDA)")
    st.write("Identifie les sujets principaux (LDA) dans un texte fourni et visualise les données d'entraînement.")
    
    if not NLTK_RESOURCES_INITIALIZED:
        st.error("Les ressources NLTK n'ont pas pu être initialisées. L'analyse de texte LDA est désactivée.")
    else:
        df_reviews_lda = load_lda_data()
        
        NUM_TOPICS_LDA_SLIDER = st.sidebar.slider("Nombre de Topics (LDA)", 2, 10, 4, key="num_topics_lda_slider")
        
        lda_pipeline_instance = None
        if df_reviews_lda is not None:
            lda_pipeline_instance = get_trained_lda_pipeline(df_reviews_lda, num_topics=NUM_TOPICS_LDA_SLIDER)
        else:
            st.error("Impossible de charger les données pour l'analyse LDA. Vérifiez le fichier CSV.")

        if lda_pipeline_instance:
            st.sidebar.success("Modèle LDA prêt.")
            
            # Afficher l'aperçu des données brutes LDA
            st.subheader("Aperçu des données brutes (LDA)")
            st.dataframe(df_reviews_lda.head())

            user_text_lda = st.text_area("Entrez un texte (en anglais) ici :", height=150, placeholder="Ex: The product was amazing...", key="text_lda_input_area")
            
            if st.button("Analyser le Sujet du Texte", key="analyze_text_lda_button"):
                if user_text_lda and user_text_lda.strip():
                    with st.spinner("Analyse LDA en cours..."):
                        topic_distribution_new = lda_pipeline_instance.transform([user_text_lda])
                        if topic_distribution_new is not None and topic_distribution_new.size > 0:
                            predicted_topic_lda_idx = np.argmax(topic_distribution_new, axis=1)[0]
                            probabilities_lda = topic_distribution_new[0]
                            
                            st.subheader("Résultats de l'Analyse LDA")
                            st.write(f"**Sujet (Topic) Prédit :** Topic {predicted_topic_lda_idx}")
                            st.write("**Probabilités par Sujet :**")
                            
                            actual_num_topics_in_model = lda_pipeline_instance.named_steps['topic_modeling'].lda_model.n_components
                            prob_df_lda = pd.DataFrame(probabilities_lda, index=[f"Topic {i}" for i in range(actual_num_topics_in_model)], columns=["Probabilité"])
                            st.dataframe(prob_df_lda.style.format("{:.2%}"))
                            
                            # Afficher les mots clés du topic prédit
                            topic_modeler_step_lda = lda_pipeline_instance.named_steps.get('topic_modeling')
                            if topic_modeler_step_lda and topic_modeler_step_lda.lda_model and topic_modeler_step_lda.count_vectorizer:
                                vectorizer_lda = topic_modeler_step_lda.count_vectorizer
                                lda_model_instance = topic_modeler_step_lda.lda_model
                                if hasattr(vectorizer_lda, 'vocabulary_') and vectorizer_lda.vocabulary_:
                                    feature_names_lda = vectorizer_lda.get_feature_names_out()
                                    n_top_words_display = 10
                                    st.write(f"**Mots clés principaux pour le Topic {predicted_topic_lda_idx} :**")
                                    try:
                                        if 0 <= predicted_topic_lda_idx < lda_model_instance.n_components:
                                            topic_loadings = lda_model_instance.components_[predicted_topic_lda_idx]
                                            top_words_indices = topic_loadings.argsort()[:-n_top_words_display - 1:-1]
                                            top_words = [feature_names_lda[i] for i in top_words_indices if i < len(feature_names_lda)]
                                            st.info(", ".join(top_words))
                                        else: st.warning(f"Index de topic prédit ({predicted_topic_lda_idx}) invalide.")
                                    except Exception as e: st.warning(f"Erreur récupération mots clés LDA: {e}")
                                else: st.warning("Vectorizer LDA non entraîné, mots clés indisponibles.")
                            
                            st.subheader("Word Clouds des Sujets LDA")
                            plot_wordcloud_lda(lda_pipeline_instance, predicted_topic_idx=predicted_topic_lda_idx)
                        else:
                            st.error("La prédiction du topic LDA a échoué.")
                else:
                    st.warning("Veuillez entrer du texte pour l'analyse LDA.")
            
            with st.expander("Afficher la visualisation t-SNE des données d'entraînement LDA"):
                tsne_vectors_lda_train, dominant_topics_lda_train = get_tsne_lda_results(lda_pipeline_instance)
                if tsne_vectors_lda_train is not None and dominant_topics_lda_train is not None:
                    actual_num_topics_lda_model = lda_pipeline_instance.named_steps['topic_modeling'].lda_model.n_components
                    plot_tsne_lda_streamlit(tsne_vectors_lda_train, dominant_topics_lda_train, actual_num_topics_lda_model)
                else:
                    st.info("La visualisation t-SNE pour les données d'entraînement LDA n'est pas disponible (calcul échoué ou données manquantes).")
        else:
            st.warning("Le pipeline LDA n'a pas pu être initialisé. Vérifiez les données et les logs.")

    st.sidebar.header("ℹ️ Infos Analyse Texte (LDA)")
    st.sidebar.info(f"""
        Utilise NLTK pour le preprocessing et Scikit-learn LDA pour la modélisation de sujets.
        Données d'entraînement LDA chargées depuis : `{LDA_DATA_FILEPATH}`.
    """)

elif analysis_type == "Analyse d'Image (Classification)":
    st.header("🖼️ Analyse d'Image")
    st.write("Prédit le \"topic\" (classe) d'une image en utilisant un pipeline pré-entraîné.")
    
    # Vérifier si les modèles d'image sont chargés
    models_loaded_check = all([feature_extractor_image_model, scaler_img_analysis, pca_img_analysis, classifier_img_analysis])
    if not models_loaded_check:
        st.error("Un ou plusieurs modèles/transformateurs pour l'analyse d'image n'ont pas pu être chargés. La prédiction est désactivée.")
    
    uploaded_file_img = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], key="image_uploader", disabled=not models_loaded_check)

    if uploaded_file_img is not None and models_loaded_check:
        pil_image = Image.open(uploaded_file_img)
        st.image(pil_image, caption="Image téléchargée", use_column_width=True)
        
        opencv_image_rgb = np.array(pil_image)
        opencv_image_bgr = cv2.cvtColor(opencv_image_rgb, cv2.COLOR_RGB2BGR)

        if st.button("Prédire le Topic de l'Image", key="predict_image_button"):
            with st.spinner("Prédiction d'image en cours..."):
                prediction_img = predict_image_topic(
                    opencv_image_bgr, 
                    feature_extractor_image_model, 
                    scaler_img_analysis, 
                    pca_img_analysis, 
                    classifier_img_analysis, 
                    label_encoder_obj=label_encoder_img_analysis
                )
                st.subheader(f"🔍 Topic Prédit (Image) : **{prediction_img}**")

    st.sidebar.header("ℹ️ Infos Analyse Image")
    st.sidebar.info(f"""
        Utilise EfficientNetB0, StandardScaler, PCA, et RandomForest.
        Modèles chargés depuis : `{SCALER_PATH}`, `{PCA_PATH}`, `{MODEL_PATH}`.
        Encodeur d'étiquettes (optionnel) : `{LABEL_ENCODER_PATH}`.
    """)
    st.sidebar.header("📊 Visualisation t-SNE (Image)")
    if os.path.exists(TSNE_PLOT_IMAGE_PATH):
        try: st.sidebar.image(TSNE_PLOT_IMAGE_PATH, caption="t-SNE des caractéristiques images (entraînement)")
        except Exception as e: st.sidebar.error(f"Erreur chargement image t-SNE: {e}")
    else:
        st.sidebar.warning(f"Image t-SNE ({TSNE_PLOT_IMAGE_PATH}) non trouvée. Exécutez le script d'entraînement pour la générer.")

st.markdown("---")
st.caption("Application développée avec Streamlit et diverses bibliothèques Python.")
