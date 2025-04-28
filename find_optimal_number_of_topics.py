import gensim
# from gensim.models import CoherenceModel # Plus nécessaire ici directement
import matplotlib.pyplot as plt
import logging
# Assurez-vous que corpora est importé si vous utilisez l'ancienne get_Cv ailleurs
# import gensim.corpora as corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ==============================================================================
# DÉFINITION DE LA FONCTION DE COHÉRENCE (Version adaptée pour Gensim)
# ==============================================================================
from gensim.models import CoherenceModel # Nécessaire pour get_Cv_gensim

def get_Cv_gensim(model, texts, dictionary, coherence_type='c_v'):
    """
    Calcule et renvoie le score de cohérence pour un modèle LDA Gensim.
    (Définition complète comme ci-dessus)
    """
    if not texts or not dictionary or not model:
        logging.error("Modèle, Textes ou Dictionnaire manquant pour le calcul de cohérence.")
        return None
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        coherence = coherence_model.get_coherence()
        return coherence
    except Exception as e:
        logging.error(f"Erreur lors du calcul de la cohérence ({coherence_type}) dans get_Cv_gensim: {e}")
        return None
# ==============================================================================


def find_optimal_number_of_topics(dictionary, corpus, texts, limit, start=2, step=3, random_state=100, passes=10, iterations=50, chunksize=100):
    """
    Calcule les scores de cohérence en utilisant get_Cv_gensim pour différents nombres de topics
    et retourne le nombre de topics optimal.

    Args:
        dictionary (gensim.corpora.Dictionary): Le dictionnaire Gensim.
        corpus (list): Le corpus au format BoW Gensim.
        texts (list of list of str): La liste des textes prétraités (liste de listes de tokens),
                                     utilisée par get_Cv_gensim.
        limit (int): Le nombre maximum de topics à tester (exclusif).
        start (int, optional): Le nombre minimum de topics à tester. Défaut à 2.
        step (int, optional): L'incrément entre les nombres de topics testés. Défaut à 3.
        random_state (int, optional): Graine pour la reproductibilité. Défaut à 100.
        passes (int, optional): Nombre de passes LDA. Défaut à 10.
        iterations (int, optional): Nombre d'itérations LDA. Défaut à 50.
        chunksize (int, optional): Taille des chunks LDA. Défaut à 100.

    Returns:
        tuple: (optimal_num_topics, coherence_scores)
    """
    coherence_values = []
    model_list = []
    topic_numbers = list(range(start, limit, step))

    if not topic_numbers:
        logging.warning(f"La plage de topics spécifiée (start={start}, limit={limit}, step={step}) est vide.")
        return None, {}

    logging.info(f"Début du calcul de cohérence (via get_Cv_gensim) pour {len(topic_numbers)} modèles (topics de {start} à {limit-1} par pas de {step})...")

    for num_topics in topic_numbers:
        try:
            # Entraînement du modèle LDA Gensim
            model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=dictionary, # Utilise le dictionnaire Gensim
                num_topics=num_topics,
                random_state=random_state,
                update_every=1,
                chunksize=chunksize,
                passes=passes,
                iterations=iterations,
                alpha='auto',
                eta='auto',
                per_word_topics=True
            )
            model_list.append(model)

            # --- MODIFICATION ICI ---
            # Calcul du score de cohérence en utilisant notre fonction adaptée get_Cv_gensim
            logging.debug(f"Appel de get_Cv_gensim pour {num_topics} topics...")
            coherence = get_Cv_gensim(model=model, texts=texts, dictionary=dictionary) # Appel de la fonction adaptée
            # --- FIN DE LA MODIFICATION ---

            if coherence is not None:
                coherence_values.append(coherence)
                logging.info(f"Nombre de Topics = {num_topics} -> Score de Cohérence (get_Cv_gensim) = {coherence:.4f}")
            else:
                logging.warning(f"get_Cv_gensim a retourné None pour {num_topics} topics. Ce point sera ignoré.")
                coherence_values.append(None)

        except Exception as e:
            # Capture d'autres erreurs potentielles (entraînement LDA, etc.)
            logging.error(f"Erreur lors de l'entraînement ou du calcul de cohérence pour {num_topics} topics: {e}")
            coherence_values.append(None)

    # Filtrer les éventuels échecs (None) avant de chercher le max
    valid_scores = [(topic_numbers[i], score) for i, score in enumerate(coherence_values) if score is not None]

    if not valid_scores:
        logging.warning("Aucun score de cohérence n'a pu être calculé avec succès via get_Cv_gensim.")
        return None, {}

    # Trouver le nombre de topics avec le score maximal
    optimal_num_topics, max_coherence = max(valid_scores, key=lambda item: item[1])
    coherence_scores_dict = dict(valid_scores)

    logging.info(f"Calcul terminé. Nombre optimal de topics trouvé : {optimal_num_topics} (Score get_Cv_gensim = {max_coherence:.4f})")

    return optimal_num_topics, coherence_scores_dict

# --- Exemple d'utilisation ---

# --- ÉTAPES PRÉALABLES (Exemple) ---
# (Assurez-vous que lemmatized_texts, dictionary, corpus sont définis comme avant)
lemmatized_texts = [
    ['ordinateur', 'rapide', 'logiciel', 'installation'],
    ['programmation', 'python', 'code', 'erreur', 'debug'],
    ['analyse', 'donnees', 'graphique', 'visualisation', 'rapport'],
    ['ordinateur', 'lent', 'virus', 'nettoyage', 'logiciel'],
    ['python', 'librairie', 'installation', 'erreur', 'code'],
    ['donnees', 'rapport', 'analyse', 'statistique', 'graphique']
]
dictionary = gensim.corpora.Dictionary(lemmatized_texts)
corpus = [dictionary.doc2bow(text) for text in lemmatized_texts]
# --- FIN DES ÉTAPES PRÉALABLES ---


# --- Utilisation de la fonction ---
start_topics = 2
limit_topics = 6
step_topics = 1

# Appeler la fonction (qui utilise maintenant get_Cv_gensim en interne)
optimal_topics, coherence_scores = find_optimal_number_of_topics(
    dictionary=dictionary,
    corpus=corpus,
    texts=lemmatized_texts, # C'est ce qui sera passé à get_Cv_gensim
    limit=limit_topics,
    start=start_topics,
    step=step_topics,
    passes=5,
    iterations=30
)

# Afficher les résultats
if optimal_topics is not None:
    print(f"\nLe nombre optimal de topics (basé sur get_Cv_gensim) est : {optimal_topics}")
    print("Scores de cohérence (get_Cv_gensim) par nombre de topics :")
    for topics, score in coherence_scores.items():
        print(f"  {topics} topics : {score:.4f}")

    # Optionnel : Visualiser les scores de cohérence
    try:
        x = list(coherence_scores.keys())
        y = list(coherence_scores.values())
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o')
        plt.xlabel("Nombre de Topics")
        plt.ylabel("Score de Cohérence (get_Cv_gensim)")
        plt.title("Évolution du Score de Cohérence (get_Cv_gensim) en fonction du Nombre de Topics")
        plt.xticks(x)
        plt.grid(True)
        plt.plot(optimal_topics, coherence_scores[optimal_topics], marker='X', color='red', markersize=10, label=f'Optimal ({optimal_topics} topics)')
        plt.legend()
        plt.show()
    except ImportError:
        print("\nMatplotlib non trouvé. Installez-le (`pip install matplotlib`) pour visualiser les scores.")
    except Exception as e:
        print(f"\nErreur lors de la création du graphique : {e}")
else:
    print("\nImpossible de déterminer le nombre optimal de topics avec les paramètres fournis (en utilisant get_Cv_gensim).")

