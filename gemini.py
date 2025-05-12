import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Pour sauvegarder les modèles et les données
import streamlit as st  # Pour l'application Streamlit

# 1. Chargement des données
def charger_donnees(chemin_fichier):
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        chemin_fichier (str): Le chemin du fichier CSV.

    Returns:
        pandas.DataFrame: Les données chargées, ou None en cas d'erreur.
    """
    try:
        df = pd.read_csv(chemin_fichier)
        print(f"Données chargées depuis : {chemin_fichier}")
        return df
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à l'emplacement : {chemin_fichier}")
        return None
    except Exception as e:
        print(f"Une erreur s'est produite lors du chargement du fichier : {e}")
        return None

# 2. Prétraitement des données
def pretraiter_donnees(df):
    """
    Prétraite les données en supprimant les valeurs manquantes et en séparant les caractéristiques (X) et la cible (y).

    Args:
        df (pandas.DataFrame): Les données à prétraiter.

    Returns:
        tuple: X et y, ou (None, None) en cas d'erreur.
    """
    if df is None:
        return None, None

    try:
        df = df.dropna()  # Supprime les lignes avec des valeurs manquantes
        X = df['text']  # Caractéristiques (texte)
        y = df['topic']  # Cible (topic)
        print("Données prétraitées.")
        return X, y
    except KeyError as e:
        print(f"Erreur : Colonne manquante dans le DataFrame : {e}")
        return None, None
    except Exception as e:
        print(f"Une erreur s'est produite lors du prétraitement des données : {e}")
        return None, None

# 3. Division des données
def diviser_donnees(X, y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        X (pandas.Series): Les caractéristiques.
        y (pandas.Series): La cible.
        test_size (float, optional): La proportion de l'ensemble de test. Par défaut, 0.2.
        random_state (int, optional): La graine pour la reproductibilité. Par défaut, 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test, ou (None,) * 4 en cas d'erreur.
    """
    if X is None or y is None:
        return None, None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Données divisées en ensembles d'entraînement et de test.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Une erreur s'est produite lors de la division des données : {e}")
        return None, None, None, None

# 4. Vectorisation du texte
def vectoriser_texte(X_train, X_test, max_features=5000):
    """
    Vectorise le texte à l'aide de TF-IDF.

    Args:
        X_train (pandas.Series): Le texte d'entraînement.
        X_test (pandas.Series): Le texte de test.
        max_features (int, optional): Nombre maximal de caractéristiques. Par défaut, 5000.

    Returns:
        tuple: Vecteur TF-IDF entraîné, X_train_vect, X_test_vect, ou (None,) * 3 en cas d'erreur.
    """
    try:
        vectoriser = TfidfVectorizer(max_features=max_features)
        X_train_vect = vectoriser.fit_transform(X_train)
        X_test_vect = vectoriser.transform(X_test)
        print("Texte vectorisé avec TF-IDF.")
        return vectoriser, X_train_vect, X_test_vect
    except Exception as e:
        print(f"Une erreur s'est produite lors de la vectorisation du texte : {e}")
        return None, None, None

# 5. Entraînement du modèle
def entrainer_modele(X_train_vect, y_train, alpha=1.0):
    """
    Entraîne un modèle Naive Bayes Multinomial.

    Args:
        X_train_vect (scipy.sparse.csr_matrix): Les caractéristiques d'entraînement vectorisées.
        y_train (pandas.Series): La cible d'entraînement.
        alpha (float, optional): Paramètre de lissage. Par défaut, 1.0.

    Returns:
        sklearn.naive_bayes.MultinomialNB: Le modèle entraîné, ou None en cas d'erreur.
    """
    try:
        modele = MultinomialNB(alpha=alpha)
        modele.fit(X_train_vect, y_train)
        print("Modèle Naive Bayes Multinomial entraîné.")
        return modele
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'entraînement du modèle : {e}")
        return None

# 6. Évaluation du modèle
def evaluer_modele(modele, X_test_vect, y_test):
    """
    Évalue le modèle entraîné.

    Args:
        modele (sklearn.naive_bayes.MultinomialNB): Le modèle entraîné.
        X_test_vect (scipy.sparse.csr_matrix): Les caractéristiques de test vectorisées.
        y_test (pandas.Series): La cible de test.

    Returns:
        tuple: accuracy, rapport de classification, ou (None, None) en cas d'erreur.
    """
    if modele is None:
        return None, None

    try:
        y_pred = modele.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        rapport_classification = classification_report(y_test, y_pred)
        print("Modèle évalué.")
        return accuracy, rapport_classification
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'évaluation du modèle : {e}")
        return None, None

# 7. Sauvegarde des artefacts
def sauvegarder_artefacts(modele, vectoriser, X_train, y_train, chemin_modele="modele.joblib", chemin_vectoriser="vectoriser.joblib", chemin_entrainement="entrainement.joblib"):
    """
    Sauvegarde le modèle entraîné, le vectoriseur et les données d'entraînement.

    Args:
        modele (sklearn.naive_bayes.MultinomialNB): Le modèle entraîné.
        vectoriser (sklearn.feature_extraction.text.TfidfVectorizer): Le vectoriseur TF-IDF.
        X_train (pandas.Series): Les caractéristiques d'entraînement.
        y_train (pandas.Series): La cible d'entraînement.
        chemin_modele (str, optional): Le chemin de sauvegarde du modèle. Par défaut, "modele.joblib".
        chemin_vectoriser (str, optional): Le chemin de sauvegarde du vectoriseur. Par défaut, "vectoriser.joblib".
        chemin_entrainement (str, optional): Le chemin de sauvegarde des données d'entraînement. Par défaut, "entrainement.joblib".
    """
    try:
        joblib.dump(modele, chemin_modele)
        joblib.dump(vectoriser, chemin_vectoriser)
        joblib.dump({'X_train': X_train, 'y_train': y_train}, chemin_entrainement) #Sauvegarde des données d'entrainement
        print(f"Modèle, vectoriseur et données d'entraînement sauvegardés.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de la sauvegarde des artefacts : {e}")

# 8. Fonction principale
def main(chemin_fichier="data/movie_review.csv"): #chemin_fichier par défaut
    """
    Fonction principale qui exécute le pipeline de traitement des données, d'entraînement du modèle et de l'évaluation.
    """
    df = charger_donnees(chemin_fichier)
    if df is None:
        return

    X, y = pretraiter_donnees(df)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = diviser_donnees(X, y)
    if None in (X_train, X_test, y_train, y_test):
        return

    vectoriser, X_train_vect, X_test_vect = vectoriser_texte(X_train, X_test)
    if None in (vectoriser, X_train_vect, X_test_vect):
        return

    modele = entrainer_modele(X_train_vect, y_train)
    if modele is None:
        return

    accuracy, rapport_classification = evaluer_modele(modele, X_test_vect, y_test)
    if accuracy is None or rapport_classification is None:
        return

    print(f"Précision : {accuracy:.2f}")
    print("Rapport de classification :")
    print(rapport_classification)

    sauvegarder_artefacts(modele, vectoriser, X_train, y_train)  # Sauvegarde des artefacts

# 9. Création de l'application Streamlit
def creer_application_streamlit():
    """
    Crée une application Streamlit pour prédire le topic d'un texte ou d'une image.
    """
    st.title("Prédiction de Topic")

    # Chargement des artefacts
    try:
        modele = joblib.load("modele.joblib")
        vectoriser = joblib.load("vectoriser.joblib")
        entrainement_data = joblib.load("entrainement.joblib") #chargement des données d'entrainement
        X_train = entrainement_data['X_train']
        y_train = entrainement_data['y_train']
    except Exception as e:
        st.error(f"Erreur : Impossible de charger les artefacts : {e}")
        return

    # Ajout de la possibilité de traiter des images
    input_type = st.radio("Type d'entrée", ["Texte", "Image"])

    if input_type == "Texte":
        texte_input = st.text_area("Entrez votre texte ici :", "C'est un super film !")
        if st.button("Prédire le topic du texte"):
            # Prédiction du topic
            texte_vect = vectoriser.transform([texte_input])
            topic_predit = modele.predict(texte_vect)[0]
            st.success(f"Le topic prédit pour le texte est : {topic_predit}")

    elif input_type == "Image":
        image_file = st.file_uploader("Téléchargez une image", type=["png", "jpg", "jpeg"])
        if image_file is not None:
            st.image(image_file, caption="Image téléchargée", use_column_width=True)
            # A compléter : Intégrer ici le code pour traiter l'image et extraire le texte
            texte_image = "exemple de texte extrait de l'image" # Placeholder
            texte_vect = vectoriser.transform([texte_image])
            topic_predit = modele.predict(texte_vect)[0]
            st.success(f"Le topic prédit pour le texte extrait de l'image est : {topic_predit}")
            st.warning("La fonctionnalité de prédiction à partir d'une image n'est pas encore totalement implémentée. Un texte d'exemple est utilisé pour la démonstration.")

    # Afficher les données d'entraînement
    if st.checkbox("Afficher les données d'entraînement"):
        st.write("Données d'entraînement (10 premières lignes) :")
        st.dataframe(pd.DataFrame({'text': X_train, 'topic': y_train}).head(10))

    # Afficher les topics possibles
    topics_uniques = y_train.unique()
    if st.checkbox("Afficher les topics possibles"):
        st.write("Topics possibles :")
        st.write(topics_uniques)

if __name__ == "__main__":
    # Exécution du pipeline principal
    main()
    # Lancement de l'application Streamlit
    creer_application_streamlit()
