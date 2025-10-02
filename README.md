# Recommandation de critiques SensCritique

Ce projet est une implémentation Python d’un système de recommandation de critiques pour la plateforme **SensCritique**.  
L’objectif est de proposer à l’utilisateur des critiques **similaires à celle qu’il est en train de lire**.

---

## Fonctionnalités
- Chargement de plusieurs fichiers CSV de critiques (par film).
- Nettoyage automatique du HTML dans les textes des critiques.
- Vectorisation des critiques avec **TF-IDF** et utilisation de la **similarité** pour trouver les plus proches.
- Interface console permettant à l’utilisateur de :
  1. Choisir un film.
  2. Choisir une critique (via son ID).
  3. Obtenir une liste de critiques similaires avec leur score de similarité.
- Affichage intégral des critiques.

---

## Technologies utilisées
- **Python 3.9+**
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/) (stopwords en français)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) (nettoyage HTML)

---

## Données
Le projet utilise des critiques issues de SensCritique (fournies au format CSV).  
Exemples de fichiers inclus :
- `fightclub_critiques.csv`
- `interstellar_critique.csv`

Chaque CSV doit contenir au minimum les colonnes suivantes :
- `id`
- `review_content`
- `review_title`
- `username`

---

## Utilisation

### 1. Installation
Clonez ce dépôt et installez les dépendances :
```bash
git clone https://github.com/Tharindu-P/SensCritique.git
cd SensCritique
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

pip install -r requirements.txt

