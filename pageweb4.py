# Importation des libraries Streamlit et Core
import nltk.downloader
import streamlit as st
import numpy as np
import pandas as pd
import re
import os
import logging
from collections import Counter

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
from sklearn.manifold import TSNE

# Importations Visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go # Pour ajouter des traces spécifiques
from wordcloud import WordCloud

# --- Configuration et Téléchargements NLTK (Mise en cache) ---
# Utiliser cache_resource pour les téléchargements qui ne changent pas
@st.cache_resource
def download_nltk_data():
    """Télécharge les ressources NLTK nécessaires."""
    resources = ["stopwords", "words", "punkt", "wordnet", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except : # Correction: Utiliser nltk.DownloadError directement
             try: # Essayer de trouver dans les corpora
                 nltk.data.find(f"corpora/{resource}")
             except: # Sinon télécharger
                 st.info(f"Téléchargement de la ressource NLTK : {resource}")
                 nltk.download(resource, quiet=True)
    return True

# Assurer le téléchargement au démarrage
NLTK_DATA_LOADED = download_nltk_data()

# --- Initialisation des variables globales (après téléchargement) ---
if NLTK_DATA_LOADED:
    stop_words_set = set(stopwords.words("english"))
    english_words_set = set(words.words())
else:
    st.error("Impossible de charger les ressources NLTK. L'application risque de ne pas fonctionner.")
    st.stop() # Arrêter l'exécution si NLTK échoue

# --- Définition des Fonctions (identiques à votre script) ---

def get_wordnet_pos(treebank_tag):
    """Convertit les étiquettes POS de Treebank en format WordNet."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        # Par défaut, considérer comme un nom
        return wordnet.NOUN

def preprocess_text(text, english_words=english_words_set, min_len_word=3, rejoin=False):
    """Pré-traite un texte (minuscules, regex, stopwords, mots anglais, lemmatisation)."""
    # S'assurer que le texte est une chaîne
    if not isinstance(text, str):
        text = str(text) # Tenter une conversion

    text = text.lower()
    # Garder les espaces pour le split, supprimer les autres caractères non-alphabétiques
    text = re.sub("[^a-z\s]", "", text)
    # Utiliser la variable globale initialisée
    words_list = [word for word in text.split() if word not in stop_words_set]


    # Filtrer par mots anglais seulement si english_words_set est disponible et non vide
    if english_words: # Vérifie si l'ensemble n'est pas vide
        filtered_corpus = [word for word in words_list if word in english_words]
    else:
        st.warning("Le filtrage par mots anglais est désactivé (dictionnaire non chargé).")
        filtered_corpus = words_list # Ne pas filtrer si le dictionnaire n'est pas là

    # Vérifier si NLTK est chargé avant de tenter pos_tag et lemmatize
    if not NLTK_DATA_LOADED:
         st.warning("Lemmatisation désactivée (ressources NLTK non chargées).")
         final_tokens = [w for w in filtered_corpus if len(w) >= min_len_word]
         if rejoin:
             return " ".join(final_tokens)
         return final_tokens

    try:
        lemmatizer = WordNetLemmatizer()
        tagged_words = pos_tag(filtered_corpus)
        words_list = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words
        ]

        final_tokens = [w for w in words_list if len(w) >= min_len_word]
    except LookupError as e:
        # Gérer le cas où une ressource spécifique manque malgré le check initial
        st.error(f"Erreur NLTK pendant le preprocessing (POS tag/lemmatisation): {e}")
        # Retourner les mots filtrés avant lemmatisation comme fallback
        words_list = filtered_corpus
    except Exception as e:
        st.error(f"Erreur inattendue pendant le preprocessing: {e}")
        words_list = filtered_corpus # Fallback

    if rejoin:
        return " ".join(final_tokens)

    return final_tokens

# --- Classes Transformer (identiques) ---

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer Scikit-learn pour le prétraitement de texte."""
    def __init__(self, english_words, min_len_word=3, rejoin=True):
        # Utiliser les ensembles globaux chargés au début
        self.english_words = english_words_set
        self.min_len_word = min_len_word
        self.rejoin = rejoin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Gérer les éventuelles valeurs non-string dans X
        return [
            preprocess_text(str(doc), self.english_words, self.min_len_word, self.rejoin)
            for doc in X
        ]


class TopicModeler(BaseEstimator, TransformerMixin):
    """Transformer Scikit-learn pour la modélisation de sujets LDA."""
    def __init__(self, num_topics=4, max_features=1000, random_state=0):
        self.num_topics = num_topics
        self.max_features = max_features
        self.random_state = random_state
        self.lda_model = None
        self.count_vectorizer = None
        self.topic_vectors = None
        self.dtm = None # Document-Term Matrix

    def fit(self, X, y=None):
        """Entraîne le CountVectorizer et le modèle LDA."""
        st.info(f"Entraînement du modèle LDA avec {self.num_topics} topics...")
        self.count_vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, # Augmenter min_df peut aider à la stabilité
            max_features=self.max_features,
            stop_words="english" # Redondant si preprocess_text est utilisé avant, mais sans danger
        )
        try:
            # X est supposé être une liste de textes pré-traités (strings)
            self.dtm = self.count_vectorizer.fit_transform(X)

            if self.dtm.shape[0] == 0 or self.dtm.shape[1] == 0:
                 st.warning("La matrice DTM est vide après vectorisation. LDA ne peut pas être entraîné.")
                 self.lda_model = None
                 return self # Retourner self même en cas d'échec partiel

            # Vérifier si le nombre de features est suffisant pour le nombre de topics
            if self.dtm.shape[1] < self.num_topics:
                 st.warning(f"Nombre de features ({self.dtm.shape[1]}) insuffisant pour {self.num_topics} topics. Réduction du nombre de topics à {self.dtm.shape[1]}.")
                 actual_num_topics = self.dtm.shape[1]
            else:
                 actual_num_topics = self.num_topics

            if actual_num_topics < 1:
                 st.error("Aucune feature trouvée après vectorisation. LDA ne peut pas être entraîné.")
                 self.lda_model = None
                 return self

            self.lda_model = LatentDirichletAllocation(
                n_components=actual_num_topics, # Utiliser le nombre ajusté
                learning_method="online",
                random_state=self.random_state,
                max_iter=10, # Peut nécessiter ajustement
                n_jobs=-1 # Utiliser tous les CPU disponibles
            )
            self.lda_model.fit(self.dtm)
            # Calculer pour les données d'entraînement pour usage ultérieur (ex: t-SNE)
            self.topic_vectors = self.lda_model.transform(self.dtm)
            st.success("Entraînement LDA terminé.")

        except ValueError as ve:
             st.error(f"Erreur de valeur pendant l'entraînement LDA (peut-être après preprocessing ?) : {ve}")
             st.warning("Vérifiez si le preprocessing ne vide pas tous les documents.")
             self.lda_model = None # Assurer que le modèle est None
        except Exception as e:
            st.error(f"Erreur inattendue pendant l'entraînement LDA : {e}")
            self.lda_model = None
        return self

    def transform(self, X):
        """Transforme de nouveaux textes en distributions de topics."""
        if self.lda_model is None or self.count_vectorizer is None:
            st.error("Le modèle LDA ou le Vectorizer n'a pas été entraîné correctement.")
            # Retourner une forme compatible mais vide ou avec des zéros
            # Le nombre de topics pourrait être celui demandé initialement ou ajusté
            num_output_topics = self.num_topics if self.lda_model is None else self.lda_model.n_components
            return np.zeros((len(X), num_output_topics))

        try:
            # X est supposé être une liste de textes pré-traités (strings)
            dtm_new = self.count_vectorizer.transform(X)
            topic_vectors_new = self.lda_model.transform(dtm_new)
            return topic_vectors_new
        except Exception as e:
            st.error(f"Erreur pendant la transformation LDA : {e}")
            return np.zeros((len(X), self.lda_model.n_components))

# --- Fonctions de Chargement et Traitement (Mise en cache) ---

@st.cache_data # Cache les données retournées
def load_data(filepath="negative_reviews5000.csv"):
    """Charge les données depuis un fichier CSV."""
    try:
        # Vérifier si le fichier existe avant de lire
        if not os.path.exists(filepath):
             st.error(f"Fichier non trouvé : {filepath}. Assurez-vous qu'il est dans le bon répertoire.")
             return None
        df = pd.read_csv(filepath)
        # S'assurer que la colonne 'text' existe et ne contient pas que des NaN
        if 'text' not in df.columns:
            st.error(f"La colonne 'text' est manquante dans {filepath}")
            return None
        df = df.dropna(subset=['text'])
        if df.empty:
            st.error(f"Aucune donnée valide trouvée dans la colonne 'text' de {filepath} après suppression des NaN.")
            return None
        return df
    except pd.errors.EmptyDataError:
         st.error(f"Le fichier CSV '{filepath}' est vide.")
         return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV '{filepath}' : {e}")
        return None

@st.cache_resource # Cache l'objet pipeline entraîné
def get_trained_pipeline(data, num_topics=4):
    """Crée et entraîne le pipeline de traitement de texte."""
    if data is None or 'text' not in data.columns or data.empty:
         st.error("Données invalides ou vides fournies pour l'entraînement du pipeline.")
         return None

    # Utiliser TextPreprocessor pour le nettoyage AVANT TopicModeler
    text_pipeline = Pipeline(
        [
            (
                "preprocessing",
                # Passer l'ensemble de mots anglais chargé globalement
                TextPreprocessor(english_words=english_words_set, min_len_word=3, rejoin=True),
            ),
            ("topic_modeling", TopicModeler(num_topics=num_topics, random_state=42)) # Fixer random_state pour reproductibilité
        ]
    )

    try:
        # Entraîner sur la colonne 'text' (TextPreprocessor gère la conversion en str)
        st.write("Début de l'entraînement du pipeline (cela peut prendre un moment)...")
        # L'étape 'preprocessing' retourne une liste de strings traités
        # L'étape 'topic_modeling' prend cette liste en entrée pour fit
        text_pipeline.fit(data["text"])
        st.write("Pipeline entraîné.")

        # Vérifier si l'entraînement LDA a réussi dans TopicModeler
        topic_modeler_step = text_pipeline.named_steps.get('topic_modeling')
        if topic_modeler_step is None or topic_modeler_step.lda_model is None:
             st.error("L'étape de modélisation de sujet (LDA) dans le pipeline a échoué ou n'a pas produit de modèle.")
             return None # Retourner None si l'étape cruciale a échoué

    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du pipeline : {e}")
        # Afficher plus de détails si possible, ex: traceback
        # import traceback
        # st.error(traceback.format_exc())
        return None

    return text_pipeline

@st.cache_data # Cache les résultats du t-SNE
def get_tsne_results(_pipeline, _data):
    """Calcule les vecteurs de sujet pour les données originales et applique t-SNE."""
    if _pipeline is None or _data is None or _data.empty:
        st.warning("Pipeline ou données manquantes/vides pour t-SNE.")
        return None, None

    topic_modeler = _pipeline.named_steps.get('topic_modeling')
    if topic_modeler is None or topic_modeler.lda_model is None:
        st.warning("Modèle LDA non trouvé dans le pipeline pour t-SNE.")
        return None, None

    # Récupérer les vecteurs de sujet déjà calculés lors du fit du pipeline
    # Ces vecteurs correspondent aux données originales (_data['text']) après preprocessing et LDA
    if hasattr(topic_modeler, 'topic_vectors') and topic_modeler.topic_vectors is not None and len(topic_modeler.topic_vectors) == len(_data):
         original_topic_vectors = topic_modeler.topic_vectors
         st.info("Utilisation des topic vectors pré-calculés lors de l'entraînement pour t-SNE.")
    else:
         # Si les vecteurs ne sont pas disponibles (ce qui ne devrait pas arriver si fit a réussi),
         # on pourrait les recalculer, mais cela indique un problème potentiel.
         st.warning("Recalcul des topic vectors pour t-SNE (inattendu)...")
         # Note: pipeline.transform applique toutes les étapes, y compris preprocessing
         original_topic_vectors = _pipeline.transform(_data['text'])
         st.info("Recalcul terminé.")


    if original_topic_vectors is None or original_topic_vectors.shape[0] == 0:
        st.warning("Les vecteurs de sujet pour les données originales sont vides ou non calculés.")
        return None, None

    st.info("Calcul de la réduction t-SNE (cela peut prendre du temps)...")
    # Ajuster perplexity en fonction de la taille des données si nécessaire (doit être < n_samples)
    n_samples = original_topic_vectors.shape[0]
    perplexity_value = min(30, n_samples - 1) # Valeur sûre
    if perplexity_value < 5:
         st.warning(f"Très peu d'échantillons ({n_samples}), t-SNE pourrait être moins significatif.")
         perplexity_value = max(1, n_samples - 1) # Minimum 1 si n_samples > 1

    if n_samples <= 1:
         st.warning("t-SNE nécessite au moins 2 échantillons.")
         return None, None


    tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=300, init='pca', learning_rate='auto')
    try:
        tsne_vectors = tsne_model.fit_transform(original_topic_vectors)
        st.success("Calcul t-SNE terminé.")
    except Exception as e:
        st.error(f"Erreur lors du calcul t-SNE : {e}")
        return None, None

    dominant_topics = np.argmax(original_topic_vectors, axis=1)

    # Retourner le modèle t-SNE n'est pas utile car il n'a pas de méthode transform
    # On retourne juste les vecteurs 2D et les topics dominants
    return tsne_vectors, dominant_topics

# --- Fonctions de Visualisation Adaptées ---

def plot_tsne_streamlit(tsne_vectors, dominant_topics, num_topics, predicted_topic_highlight=None):
    """Affiche la visualisation t-SNE avec Plotly. Peut mettre en évidence les points d'un topic spécifique."""
    if tsne_vectors is None or dominant_topics is None:
        st.warning("Données t-SNE manquantes pour la visualisation.")
        return
    if len(tsne_vectors) == 0:
         st.warning("Aucun point à afficher dans le graphique t-SNE.")
         return

    st.write(f"Visualisation t-SNE des {len(tsne_vectors)} avis originaux.")

    df = pd.DataFrame(tsne_vectors, columns=['x', 'y'])
    # Assurer que dominant_topics a la même longueur que tsne_vectors
    if len(dominant_topics) != len(tsne_vectors):
         st.error("Incohérence entre le nombre de points t-SNE et les topics dominants.")
         return
    df['dominant_topic'] = dominant_topics.astype(str)

    # Définir une palette de couleurs
    # Utiliser une palette Plotly standard qui gère plus de couleurs
    colors = px.colors.qualitative.Plotly
    # S'assurer que num_topics correspond au nombre réel de topics dans le modèle LDA
    actual_num_topics = len(set(dominant_topics))
    color_map = {str(i): colors[i % len(colors)] for i in range(actual_num_topics)}

    fig = px.scatter(df, x='x', y='y', color='dominant_topic',
                     color_discrete_map=color_map,
                     title='Visualisation t-SNE des Sujets LDA (Données Originales)',
                     labels={'dominant_topic': 'Sujet Dominant'},
                     hover_data={'dominant_topic': True, df.index.name: df.index}, # Afficher topic et index au survol
                     template='plotly_white',
                     opacity=0.7) # Rendre les points originaux légèrement transparents

    # Optionnel : Mettre en évidence les points du topic prédit pour le nouveau texte
    if predicted_topic_highlight is not None:
        highlight_topic_str = str(predicted_topic_highlight)
        # Ou plus simplement, ajouter une annotation
        st.info(f"Le topic prédit pour votre texte est le Topic {predicted_topic_highlight}. Les points correspondants dans le graphique ci-dessus représentent les avis originaux de ce même topic.")
        # On pourrait aussi changer l'apparence des points de ce topic, mais cela peut complexifier le graphique.
        # Exemple (non testé, nécessite ajustement) :
        # fig.for_each_trace(lambda t: t.update(marker=dict(size=10, symbol='diamond')) if t.name == highlight_topic_str else ())


    # Note: Le nouveau point texte n'est PAS ajouté au graphique t-SNE lui-même.
    fig.update_layout(legend_title_text='Topics')
    st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud_streamlit(pipeline, num_topics, predicted_topic=None):
    """Affiche les nuages de mots pour chaque sujet, en encadrant le sujet prédit."""
    topic_modeler = pipeline.named_steps.get('topic_modeling')
    if topic_modeler is None or topic_modeler.lda_model is None or topic_modeler.count_vectorizer is None:
        st.warning("Modèle LDA ou Vectorizer non trouvé pour générer les Word Clouds.")
        return

    lda_model = topic_modeler.lda_model
    vectorizer = topic_modeler.count_vectorizer
    # Vérifier si le vectorizer a été entraîné
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
         st.warning("Vectorizer non entraîné, impossible de récupérer les noms de features pour les Word Clouds.")
         return
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des features du vectorizer: {e}")
        return

    # Utiliser le nombre réel de composants du modèle LDA
    actual_num_topics = lda_model.n_components

    # Déterminer la disposition des subplots
    ncols = 2
    nrows = int(np.ceil(actual_num_topics / ncols))
    if nrows == 0 or ncols == 0: # Cas où actual_num_topics est 0
         st.warning("Aucun topic trouvé dans le modèle LDA.")
         return

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 5), squeeze=False)
    axes = axes.flatten()

    for i in range(actual_num_topics):
        topic_words = lda_model.components_[i]
        # Prendre le top 20 mots ou moins si moins de features
        top_word_indices = topic_words.argsort()[:-min(21, len(feature_names)):-1]
        wc_dict = {
            feature_names[index]: topic_words[index]
            for index in top_word_indices if index < len(feature_names) # Vérification supplémentaire
        }

        ax = axes[i] # Sélectionner l'axe courant

        if not wc_dict: # Si aucun mot n'est trouvé pour ce topic
             ax.set_title(f"Topic {i} (Vide)")
             ax.text(0.5, 0.5, "Aucun mot significatif trouvé", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.axis("off")
             continue

        wc = WordCloud(background_color="white", width=400, height=200, max_words=50) # Limiter max_words
        try:
            wc.generate_from_frequencies(wc_dict)
            ax.set_title(f"Topic {i}")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

            # Encadrer le topic prédit
            if predicted_topic is not None and i == predicted_topic:
                # Ajouter une bordure rouge
                rect = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='red', lw=3, transform=ax.transAxes, clip_on=False)
                ax.add_patch(rect)
                # Mettre le titre en évidence
                ax.set_title(f"Topic {i} (Votre Texte)", color='red', fontweight='bold')

        except ValueError as ve:
             st.warning(f"Erreur lors de la génération du WordCloud pour le Topic {i}: {ve}")
             ax.set_title(f"Topic {i} (Erreur WC)")
             ax.axis("off")
        except Exception as e:
             st.error(f"Erreur inattendue WordCloud Topic {i}: {e}")
             ax.set_title(f"Topic {i} (Erreur Inattendue)")
             ax.axis("off")

    # Masquer les axes restants si num_topics n'est pas un multiple de ncols
    for j in range(actual_num_topics, nrows * ncols):
         if j < len(axes):
            axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le suptitle
    plt.suptitle("Word Clouds par Sujet LDA", fontsize=16)
    st.pyplot(fig)
    # Fermer la figure pour libérer la mémoire
    plt.close(fig)


# --- Application Streamlit ---

#st.set_page_config(layout="wide") # Mettre en page large par défaut
st.title("Analyse de Sujets (Topics) LDA pour Avis Clients")

# --- Chargement et Entraînement (mis en cache) ---
data_load_state = st.text("Chargement des données...")
df_reviews = load_data() # Utilise le chemin par défaut "negative_reviews5000.csv"
if df_reviews is not None:
    data_load_state.text(f"Chargement des données... Terminé! ({len(df_reviews)} avis chargés)")
else:
    data_load_state.error("Chargement des données échoué.")
    st.stop() # Arrêter si les données ne peuvent pas être chargées

st.subheader("Aperçu des données brutes")
st.dataframe(df_reviews.head())

# Définir le nombre de topics ici
# Idéalement, ce nombre serait déterminé par une analyse préalable (ex: score de cohérence)
# Pour l'exemple, on fixe à 4
NUM_TOPICS = 4
st.sidebar.info(f"Nombre de topics configuré : {NUM_TOPICS}")

# Obtenir le pipeline entraîné (depuis le cache si possible)
pipeline = get_trained_pipeline(df_reviews, num_topics=NUM_TOPICS)

# Vérifier si le pipeline a été entraîné correctement
if pipeline is None:
    st.error("Le pipeline d'analyse n'a pas pu être entraîné. Vérifiez les logs et les données d'entrée.")
    st.stop() # Arrêter si le pipeline échoue

# Vérifier le nombre réel de topics dans le modèle entraîné
topic_modeler_step = pipeline.named_steps.get('topic_modeling')
if topic_modeler_step and topic_modeler_step.lda_model:
     ACTUAL_NUM_TOPICS = topic_modeler_step.lda_model.n_components
     if ACTUAL_NUM_TOPICS != NUM_TOPICS:
          st.sidebar.warning(f"Le nombre de topics a été ajusté à {ACTUAL_NUM_TOPICS} en raison des données.")
else:
     # Si le modèle n'est pas là, on ne peut pas continuer
     st.error("Le modèle LDA n'a pas été trouvé dans le pipeline après l'entraînement.")
     st.stop()


# Obtenir les résultats t-SNE (depuis le cache si possible)
# Passer le pipeline et les données originales
tsne_vectors, dominant_topics = get_tsne_results(pipeline, df_reviews)

st.sidebar.success("Modèle prêt pour la prédiction.")

# --- Interface Utilisateur ---
st.divider()
st.header("Tester avec votre propre texte")
user_text = st.text_area("Entrez un avis (en anglais) ici :", height=150, placeholder="Ex: The product was amazing, great quality and fast delivery!")

if st.button("Analyser le Sujet"):
    if user_text and user_text.strip(): # Vérifier si le texte n'est pas vide ou juste des espaces
        with st.spinner("Analyse en cours..."):
            # 1. Prédire la distribution des topics pour le nouveau texte
            #    Utiliser directement pipeline.transform qui applique toutes les étapes
            #    Assurer que l'entrée est une liste ou similaire
            topic_distribution_new = pipeline.transform([user_text])

            if topic_distribution_new is not None and topic_distribution_new.size > 0:
                # Trouver l'index du topic le plus probable
                predicted_topic = np.argmax(topic_distribution_new, axis=1)[0]
                probabilities = topic_distribution_new[0]

                st.subheader("Résultats de l'Analyse")
                st.write(f"**Sujet (Topic) Prédit :** Topic {predicted_topic}")
                st.write("**Probabilités par Sujet :**")
                # Créer un DataFrame pour un affichage clair des probabilités
                prob_df = pd.DataFrame(
                    probabilities,
                    index=[f"Topic {i}" for i in range(ACTUAL_NUM_TOPICS)], # Utiliser le nombre réel de topics
                    columns=["Probabilité"]
                )
                st.dataframe(prob_df.style.format("{:.2%}")) # Afficher en pourcentage

                # Afficher les mots clés du topic prédit
                # Récupérer le modèle et le vectorizer depuis le pipeline
                topic_modeler = pipeline.named_steps.get('topic_modeling')
                if topic_modeler and topic_modeler.lda_model and topic_modeler.count_vectorizer:
                    vectorizer = topic_modeler.count_vectorizer
                    lda_model = topic_modeler.lda_model
                    # Vérifier si le vectorizer a des features
                    if hasattr(vectorizer, 'vocabulary_') and vectorizer.vocabulary_:
                        feature_names = vectorizer.get_feature_names_out()
                        n_top_words = 10 # Nombre de mots clés à afficher
                        st.write(f"**Mots clés principaux pour le Topic {predicted_topic} :**")
                        try:
                            # S'assurer que predicted_topic est un index valide
                            if 0 <= predicted_topic < lda_model.n_components:
                                topic_loadings = lda_model.components_[predicted_topic]
                                top_words_indices = topic_loadings.argsort()[:-n_top_words - 1:-1]
                                # Vérifier les indices avant d'accéder à feature_names
                                top_words = [
                                    feature_names[i] for i in top_words_indices if i < len(feature_names)
                                ]
                                st.info(", ".join(top_words))
                            else:
                                st.warning(f"Index de topic prédit ({predicted_topic}) invalide.")
                        except IndexError:
                             st.warning(f"Impossible de récupérer les mots clés pour le topic {predicted_topic} (Index hors limites).")
                        except Exception as e:
                             st.warning(f"Erreur lors de la récupération des mots clés : {e}")
                    else:
                         st.warning("Vectorizer non entraîné, impossible d'afficher les mots clés.")
                else:
                     st.warning("Impossible de récupérer le modèle LDA ou le Vectorizer depuis le pipeline.")


                st.divider()
                st.subheader("Visualisations")

                # 2. Visualisation t-SNE (sans projeter le nouveau point)
                if tsne_vectors is not None and dominant_topics is not None:
                    # Passer le topic prédit pour mise en évidence potentielle
                    try:
                        plot_tsne_streamlit(tsne_vectors, dominant_topics, ACTUAL_NUM_TOPICS, predicted_topic_highlight=predicted_topic)
                    except AttributeError as ae: # Capture l'erreur spécifique si on essaie encore d'utiliser transform (ne devrait plus arriver)
                         st.warning(f"Impossible de transformer le nouveau point pour t-SNE (peut-être dimensions incompatibles): {ve}")
                    except Exception as e:
                         st.error(f"Erreur lors de la projection t-SNE du nouveau point : {e}")

                else:
                    st.warning("Visualisation t-SNE non disponible (calcul échoué ou données manquantes).")

                # 3. Visualisation Word Cloud en mettant en évidence le topic prédit
                plot_wordcloud_streamlit(pipeline, ACTUAL_NUM_TOPICS, predicted_topic=predicted_topic)

            else:
                st.error("La prédiction du topic a échoué pour le texte fourni. Vérifiez le preprocessing ou le modèle.")
    else:
        st.warning("Veuillez entrer du texte à analyser.")


st.sidebar.markdown("---")
st.sidebar.header("Informations")
st.sidebar.markdown("""
Cette application utilise le modèle LDA (Latent Dirichlet Allocation) pour identifier
les sujets cachés dans un corpus d'avis clients négatifs.

- Entrez un nouvel avis pour voir quel sujet lui est attribué.
- La visualisation t-SNE montre la proximité des documents originaux en fonction de leurs sujets.
- Les Word Clouds affichent les mots les plus représentatifs de chaque sujet.
""")

# Optionnel: Ajouter un pied de page
st.markdown("---")
st.caption("Application développée avec Streamlit, NLTK, Scikit-learn, Plotly, Matplotlib.")
