# Victory Vision Analytics (VVA)

Prédicteur de podium de Grands Prix de Formule 1 par machine learning.

VVA analyse les données historiques de course pour calculer la probabilité de podium de chaque pilote avant un Grand Prix, à partir de la grille de départ.

## Fonctionnement

1. Saisir la grille de départ (pilote + position) et le circuit
2. Le modèle calcule pour chaque pilote sa probabilité d'atteindre le podium
3. Les 3 candidats les plus probables sont affichés

## Modèle

Le pipeline ML est entraîné sur les données F1 depuis 2022, avec les features suivantes :

- Position sur la grille de départ
- Expérience du pilote et du constructeur (nombre de courses cumulées)
- Victoires cumulées pilote et constructeur
- Expérience du pilote avec son constructeur actuel
- Âge du pilote au moment de la course
- Score de fiabilité du constructeur (DNF score)
- Longueur et nombre de virages du circuit
- Position au dernier Grand Prix

Trois modèles ont été comparés par cross-validation (Random Forest, SVM, KNN) — Random Forest obtient les meilleurs résultats et est utilisé en production.

## Stack

Python · scikit-learn · pandas · NumPy · Streamlit · BeautifulSoup · joblib · Plotly

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Construire les données et entraîner le modèle
python main.py

# Lancer l'interface
streamlit run dev.py
```

## Structure

```
VVA/
├── main.py              # Point d'entrée
├── data_loader.py       # Chargement CSV + scraping Wikipedia
├── preprocessing.py     # Nettoyage et traitement des données
├── features.py          # Feature engineering
├── model.py             # Entraînement et évaluation
├── predict.py           # Fonction de prédiction
├── dev.py               # Dashboard Streamlit
├── model/               # Modèle sérialisé (joblib)
├── data/                # Données intermédiaires
└── F1 Championship Archive/  # Dataset historique F1
```
