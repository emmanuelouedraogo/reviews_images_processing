import streamlit as st
import cv2
import numpy as np
from PIL import Image # Pour gérer les images téléchargées par Streamlit
import joblib # Pour charger les objets scikit-learn sauvegardés
import os

# Imports TensorFlow/Keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Configuration ---
# Chemins vers les composants du modèle sauvegardés (doivent correspondre à ceux de Nmeth.py)
SCALER_PATH = 'scaler.joblib'
PCA_PATH = 'pca.joblib'
MODEL_PATH = 'random_forest_model.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib' # Optionnel, seulement si utilisé lors de l'entraînement

TSNE_PLOT_IMAGE_PATH = 'tsne_visualization.png' # Doit correspondre à Nmeth.py

EFFICIENTNET_INPUT_SIZE = (224, 224) # Doit correspondre à la taille utilisée lors de l'entraînement

# --- Chargement des Modèles et Transformateurs ---

# Extracteur de caractéristiques (EfficientNetB0) - chargé une seule fois
# Pas besoin de sauvegarder/charger spécifiquement si vous utilisez les poids 'imagenet' et non fine-tuné.
@st.cache_resource # Mise en cache pour améliorer les performances après le premier chargement
def load_feature_extractor():
    model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    return model

feature_extractor = load_feature_extractor()

# Fonction pour charger les objets scikit-learn
@st.cache_resource
def load_sklearn_object(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'objet depuis {path}: {e}")
            return None
    else:
        # Ne pas afficher d'erreur ici si le fichier est optionnel (comme label_encoder)
        # La logique de prédiction gérera si les objets essentiels manquent.
        if path != LABEL_ENCODER_PATH: # Avertir seulement pour les fichiers essentiels
             st.warning(f"Avertissement : Fichier non trouvé à {path}. La prédiction pourrait ne pas fonctionner.")
        return None

scaler = load_sklearn_object(SCALER_PATH)
pca = load_sklearn_object(PCA_PATH)
classifier = load_sklearn_object(MODEL_PATH)
label_encoder = load_sklearn_object(LABEL_ENCODER_PATH) # Sera None si le fichier n'existe pas ou n'a pas été utilisé

# --- Fonction de Prédiction ---
def predict_image_topic(image_array_bgr, feature_extractor_model, scaler_obj, pca_obj, classifier_model, label_encoder_obj=None):
    """
    Prédit le topic d'une image donnée (attend un array OpenCV BGR).
    """
    if image_array_bgr is None:
        return "Erreur: Image non valide."

    try:
        # Prétraitement de l'image (identique à celui de l'entraînement)
        img_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB) # Conversion en RGB pour EfficientNet
        img_resized = cv2.resize(img_rgb, EFFICIENTNET_INPUT_SIZE)
        img_expanded = np.expand_dims(img_resized, axis=0) # Ajout de la dimension du batch
        img_preprocessed = preprocess_input(img_expanded) # Prétraitement spécifique à EfficientNet

        # 1. Extraction des caractéristiques
        features_new_image = feature_extractor_model.predict(img_preprocessed, verbose=0)[0]
        features_new_image_reshaped = features_new_image.reshape(1, -1) # Pour scaler et pca

        # Vérification que les composants essentiels sont chargés
        if scaler_obj is None: return "Erreur: Scaler non chargé. Impossible de prédire."
        if pca_obj is None: return "Erreur: PCA non chargé. Impossible de prédire."
        if classifier_model is None: return "Erreur: Modèle de classification non chargé. Impossible de prédire."

        # 2. Standardisation
        features_new_image_std = scaler_obj.transform(features_new_image_reshaped)

        # 3. Transformation PCA
        features_new_image_pca = pca_obj.transform(features_new_image_std)

        # 4. Prédiction
        prediction_numeric = classifier_model.predict(features_new_image_pca)[0]

        # 5. Décodage de l'étiquette (si un encodeur a été utilisé et est ajusté)
        if label_encoder_obj and hasattr(label_encoder_obj, 'classes_'):
            return label_encoder_obj.inverse_transform([prediction_numeric])[0]
        else:
            return str(prediction_numeric) # Retourne la prédiction numérique comme chaîne

    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

# --- Interface Utilisateur Streamlit ---
#st.set_page_config(page_title="Prédiction de Topic d'Image", layout="centered")
st.title("🖼️ Prédiction de Topic d'Image")

st.write("""
    Téléchargez une image (JPG, JPEG, PNG) et l'application prédira son topic
    en utilisant un pipeline de Machine Learning pré-entraîné.
""")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Image téléchargée", use_column_width=True)
    
    # Conversion de l'image PIL (RGB) en format OpenCV (BGR) pour la fonction de prédiction
    opencv_image_rgb = np.array(pil_image)
    opencv_image_bgr = cv2.cvtColor(opencv_image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Prédiction en cours..."):
        prediction = predict_image_topic(
            opencv_image_bgr,
            feature_extractor,
            scaler,
            pca,
            classifier,
            label_encoder_obj=label_encoder
        )
        st.subheader(f"🔍 Topic Prédit : **{prediction}**")

st.sidebar.header("ℹ️ À propos")
st.sidebar.info(f"""
    Cette application utilise EfficientNetB0 pour l'extraction de caractéristiques, suivi d'un StandardScaler, d'une PCA, et d'un classifieur RandomForest.
    Les modèles et transformateurs scikit-learn sont chargés depuis :
    - `{SCALER_PATH}`
    - `{PCA_PATH}`
    - `{MODEL_PATH}`
    - `{LABEL_ENCODER_PATH}` (si utilisé)
""")
st.sidebar.header("📊 Visualisation t-SNE")
if os.path.exists(TSNE_PLOT_IMAGE_PATH):
    try:
        st.sidebar.image(TSNE_PLOT_IMAGE_PATH, caption="Visualisation t-SNE des caractéristiques (entraînement)")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement de l'image t-SNE: {e}")
else:
    st.sidebar.warning(f"L'image de la visualisation t-SNE ({TSNE_PLOT_IMAGE_PATH}) n'a pas été trouvée. Veuillez exécuter le script d'entraînement pour la générer.")
