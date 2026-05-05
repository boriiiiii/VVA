from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump


def train_model(formula_1):
    """Train a RandomForest pipeline on formula_1 and return it with the train/test split."""
    # Drop non-feature columns
    X = formula_1.drop(['position', 'seconds', 'podium', 'date', 'fastestLapSpeed', 'raceId'], axis=1)
    y = formula_1['podium']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compare candidate models via 5-fold cross-validation before choosing one
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = SVC(random_state=42)
    clf3 = KNeighborsClassifier()

    scores = {}
    for model in [clf1, clf2, clf3]:
        score = cross_val_score(model, X, y, cv=5).mean()
        scores[model.__class__.__name__] = score

    for model_name, score in scores.items():
        print(f"{model_name}: {score:.2f}")

    # Final pipeline using RandomForest (best-performing in practice)
    formula1_predict = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    formula1_predict.fit(X, y)
    return formula1_predict, X, y, X_test, y_test


def evaluate_model(formula1_predict, X, y, X_test, y_test):
    """Print the average 5-fold cross-validation score for the trained model."""
    rfc_cv = cross_val_score(formula1_predict, X, y, cv=5)
    print(f"Average cross-validation score: {rfc_cv.mean() * 100:.2f}%")


def save_model(formula1_predict, path='model/formula1_model.joblib'):
    """Persist the trained model pipeline to disk with joblib."""
    dump(formula1_predict, path)
