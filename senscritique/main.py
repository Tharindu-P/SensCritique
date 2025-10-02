import re
import pandas as pd
from typing import Union, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup  # Pour nettoyer le HTML

# Télécharge automatiquement les stopwords français si besoin
try:
    french_stopwords = stopwords.words("french")
except LookupError:
    nltk.download("stopwords")
    french_stopwords = stopwords.words("french")


class ReviewRecommender:
    def __init__(self, csv_paths: Union[str, List[Tuple[str, str]]], default_film_name: str = None):
        """
        csv_paths peut être :
          - un chemin str vers un CSV (dans ce cas default_film_name peut définir la colonne 'film')
          - une liste pour concaténer plusieurs fichiers
        """
        # Charger les critiques
        if isinstance(csv_paths, (list, tuple)):
            dfs = []
            for item in csv_paths:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    raise ValueError("Si csv_paths est une liste, chaque élément doit être (path, film_name).")
                path, film_name = item
                df = pd.read_csv(path)
                df["film"] = film_name
                dfs.append(df)
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_csv(csv_paths)
            # Si l'appelant a donné un nom de film, l'appliquer
            if default_film_name is not None:
                self.df["film"] = default_film_name
            # sinon, si la colonne 'film' n'existe pas, on la crée avec une valeur par défaut
            elif "film" not in self.df.columns:
                self.df["film"] = "unknown"

        # Vérifier les colonnes
        required_cols = ["id", "review_content", "review_title", "username", "film"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Colonne manquante dans le CSV : {col}")

        # Nettoyer
        self.df["review_content"] = self.df["review_content"].fillna("").apply(self.clean_html)

        # TF-IDF vectorisation sur tout le dataset
        self.vectorizer = TfidfVectorizer(stop_words=french_stopwords)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["review_content"])

    @staticmethod
    def clean_html(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # On ne passe à BeautifulSoup si il y a des balises HTML
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")
        # remplace les espaces et sauts de ligne par un seul espace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_similar_reviews(self, review_id: int, top_n: int = 5):
        """
        Renvoie les critiques les plus similaires à la critique donnée
        """
        # Vérifier que l'ID existe
        if review_id not in self.df["id"].values:
            raise ValueError(f"L'id {review_id} n'existe pas dans le dataset")

        # Position de la critique dans le DataFrame
        idx = int(self.df.index[self.df["id"] == review_id][0])
        film_of_review = self.df.at[idx, "film"]

        # Indices des critiques du même film
        same_film_mask = self.df["film"] == film_of_review
        same_film_indices = self.df.index[same_film_mask].tolist()

        # Si il n'y a qu'une seule critique pour ce film, rien à proposer
        if len(same_film_indices) <= 1:
            return pd.DataFrame(columns=["id", "review_content", "username", "review_title", "similarity_score"])

        # Calculer les similarités seulement entre la critique choisie et les critiques du même film
        subset_matrix = self.tfidf_matrix[same_film_indices]
        # Trouver la position de idx dans same_film_indices
        pos_in_subset = same_film_indices.index(idx)

        cosine_similarities = cosine_similarity(subset_matrix[pos_in_subset], subset_matrix).flatten()

        # Trier par score décroissant
        similar_order = cosine_similarities.argsort()[::-1]

        # Prendre les critiques les plus ressemblante
        similar_order = [i for i in similar_order if i != pos_in_subset][:top_n]

        # Map back to original dataframe indices
        selected_indices = [same_film_indices[i] for i in similar_order]

        # Préparer résultats
        results = self.df.loc[selected_indices, ["id", "review_content", "username", "review_title"]].copy()
        results["similarity_score"] = cosine_similarities[similar_order]

        results = results.reset_index(drop=True)
        return results


if __name__ == "__main__":
    # Afficher tout le texte complet des cellules sans troncature
    pd.set_option("display.max_colwidth", None)
    recommender = ReviewRecommender([
        ("fightclub_critiques.csv", "Fight Club"),
        ("interstellar_critique.csv", "Interstellar"),
    ])

    # Lister les films disponibles
    films = sorted(recommender.df["film"].unique().tolist())
    print("Films disponibles :")
    for i, f in enumerate(films, 1):
        print(f"{i}. {f}")

    # Choix du film
    while True:
        try:
            film_choice = int(input("\nEntrez le numéro du film à utiliser (ex: 1) : ").strip())
            if 1 <= film_choice <= len(films):
                chosen_film = films[film_choice - 1]
                break
            else:
                print("Numéro invalide, réessayez.")
        except ValueError:
            print("Entrée invalide, tapez un nombre correspondant au film.")

    # Afficher quelques critiques
    film_df = recommender.df[recommender.df["film"] == chosen_film]
    print(f"\nQuelques critiques disponibles pour '{chosen_film}' :")
    print(film_df[["id", "review_title", "username"]].head(15).to_string(index=False))

    # Demander l'ID de la critique à l'utilisateur
    while True:
        try:
            review_id = int(input("\n👉 Entrez l'ID d'une critique pour voir les plus similaires : ").strip())
            # Vérifier que l'ID existe
            if review_id not in recommender.df["id"].values:
                print("Cet ID n'existe pas dans le dataset, réessayez.")
                continue
            # Vérifier appartenance au film
            film_of_id = recommender.df.loc[recommender.df["id"] == review_id, "film"].iloc[0]
            if film_of_id != chosen_film:
                print(f"L'ID {review_id} n'appartient pas au film sélectionné ({chosen_film}). Réessayez.")
                continue
            break
        except ValueError:
            print("Entrée invalide, tapez un nombre entier correspondant à un ID.")

    # Afficher la critique choisie avant
    chosen_row = recommender.df[recommender.df["id"] == review_id].iloc[0]
    print("\n" + "=" * 80)
    print(f"Critique choisie (ID: {chosen_row['id']}) - Utilisateur: {chosen_row['username']} - Titre: {chosen_row['review_title']}")
    print("\n" + chosen_row["review_content"])
    print("\n" + "=" * 80 + "\n")

    # Récupérer et afficher similaires
    similar = recommender.get_similar_reviews(review_id, top_n=5)
    if similar.empty:
        print("Aucune autre critique disponible pour ce film.")
    else:
        print(f"Critiques similaires à la critique {review_id} (même film) :\n")
        for idx, row in similar.iterrows():
            print(f"ID: {row['id']}")
            print(f"Utilisateur: {row['username']}")
            print(f"Titre: {row['review_title']}")
            print(f"Score de similarité: {row['similarity_score']:.3f}")
            print("Critique complète:")
            print(row["review_content"])
            print("\n" + "=" * 80 + "\n")
