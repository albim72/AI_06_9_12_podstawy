# ============================================================
# DRZEWO DECYZYJNE DLA PIMA INDIANS DIABETES
# Dane bez nagłówków + obsługa zer + optymalizacja modelu
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. NAZWY KOLUMN
# ============================================================

column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]


# ============================================================
# 2. WCZYTANIE DANYCH BEZ NAGŁÓWKÓW
# ============================================================

df = pd.read_csv(
    "diabetes.csv",
    header=None,
    names=column_names
)

print("\nPierwsze rekordy:")
print(df.head())

print("\nInformacje o danych:")
print(df.info())

print("\nStatystyki opisowe:")
print(df.describe())

print("\nRozkład klas:")
print(df["Outcome"].value_counts())


# ============================================================
# 3. KOLUMNY, W KTÓRYCH ZERO OZNACZA PRAWDOPODOBNIE BRAK DANYCH
# ============================================================

zero_as_missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\nLiczba zer przed czyszczeniem:")

for col in zero_as_missing_columns:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count}")


# ============================================================
# 4. ZAMIANA PODEJRZANYCH ZER NA NaN
# ============================================================

df_clean = df.copy()

for col in zero_as_missing_columns:
    df_clean[col] = df_clean[col].replace(0, np.nan)

print("\nLiczba braków danych po zamianie zer na NaN:")
print(df_clean.isna().sum())


# ============================================================
# 5. PODZIAŁ NA CECHY X I ETYKIETĘ y
# ============================================================

X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]


# ============================================================
# 6. PODZIAŁ NA DANE TRENINGOWE I TESTOWE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. PIPELINE: IMPUTACJA BRAKÓW + DRZEWO DECYZYJNE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)


# ============================================================
# 8. SIATKA PARAMETRÓW DO OPTYMALIZACJI
# ============================================================

param_grid = {
    "imputer__strategy": ["mean", "median"],

    "model__criterion": ["gini", "entropy"],

    "model__max_depth": [2, 3, 4, 5, 6, 7, None],

    "model__min_samples_split": [2, 5, 10, 20],

    "model__min_samples_leaf": [1, 2, 5, 10],

    "model__class_weight": [None, "balanced"]
}


# ============================================================
# 9. OPTYMALIZACJA MODELU
# ============================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


print("\nNajlepsze parametry:")
print(grid_search.best_params_)

print("\nNajlepszy wynik F1 z walidacji krzyżowej:")
print(grid_search.best_score_)


# ============================================================
# 10. OCENA NA ZBIORZE TESTOWYM
# ============================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

print("\nRaport klasyfikacyjny:")
print(classification_report(y_test, y_pred))

print("\nMiary jakości modelu:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))


# ============================================================
# 11. WAŻNOŚĆ CECH
# ============================================================

tree_model = best_model.named_steps["model"]

feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nWażność cech:")
print(feature_importances)


plt.figure(figsize=(10, 6))
plt.bar(
    feature_importances["Feature"],
    feature_importances["Importance"]
)
plt.xticks(rotation=45)
plt.title("Ważność cech w drzewie decyzyjnym")
plt.xlabel("Cecha")
plt.ylabel("Ważność")
plt.tight_layout()
plt.show()


# ============================================================
# 12. WIZUALIZACJA DRZEWA
# ============================================================

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["No diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Zoptymalizowane drzewo decyzyjne dla Pima Indians Diabetes")
plt.show()


# ============================================================
# 13. PREDYKCJA DLA NOWEJ OSOBY
# ============================================================

new_patient = pd.DataFrame(
    [[
        3,        # Pregnancies
        140,      # Glucose
        80,       # BloodPressure
        25,       # SkinThickness
        0,        # Insulin, tu 0 będzie potraktowane jako brak danych
        32.5,     # BMI
        0.45,     # DiabetesPedigreeFunction
        42        # Age
    ]],
    columns=X.columns
)

# Zamiana zer na NaN także dla nowego pacjenta

for col in zero_as_missing_columns:
    new_patient[col] = new_patient[col].replace(0, np.nan)

prediction = best_model.predict(new_patient)
prediction_proba = best_model.predict_proba(new_patient)

print("\nPredykcja dla nowej osoby:")

if prediction[0] == 1:
    print("Wynik: podwyższone ryzyko cukrzycy")
else:
    print("Wynik: brak podwyższonego ryzyka cukrzycy")

print("Prawdopodobieństwa klas:")
print("No diabetes:", prediction_proba[0][0])
print("Diabetes:", prediction_proba[0][1])
