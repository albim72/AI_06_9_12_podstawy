import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Generujemy przykładowe dane pogodowe
# ---------------------------------------------------------

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "day_of_year": np.random.randint(1, 366, n),
    "hour": np.random.randint(0, 24, n),
    "humidity": np.random.uniform(30, 100, n),
    "pressure": np.random.uniform(980, 1035, n),
    "wind_speed": np.random.uniform(0, 15, n)
})

# Symulujemy temperaturę:
# - sezonowość roczna
# - rytm dobowy
# - wpływ wilgotności, ciśnienia i wiatru
# - losowy szum

seasonal_effect = 12 * np.sin(2 * np.pi * (data["day_of_year"] - 80) / 365)
daily_effect = 5 * np.sin(2 * np.pi * (data["hour"] - 6) / 24)

temperature = (
    10
    + seasonal_effect
    + daily_effect
    - 0.04 * data["humidity"]
    + 0.03 * (data["pressure"] - 1000)
    - 0.3 * data["wind_speed"]
    + np.random.normal(0, 2, n)
)

data["temperature"] = temperature


# ---------------------------------------------------------
# 2. Przygotowanie cech X i zmiennej docelowej y
# ---------------------------------------------------------

X = data[["day_of_year", "hour", "humidity", "pressure", "wind_speed"]]
y = data["temperature"]


# ---------------------------------------------------------
# 3. Podział na zbiór treningowy i testowy
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# 4. Budowa modelu Random Forest Regressor
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Predykcja temperatury
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 6. Ocena jakości modelu
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 3))


# ---------------------------------------------------------
# 7. Porównanie wartości prawdziwych i przewidywanych
# ---------------------------------------------------------

results = pd.DataFrame({
    "Real temperature": y_test.values,
    "Predicted temperature": y_pred
})

print(results.head(10))


# ---------------------------------------------------------
# 8. Wykres: rzeczywista vs przewidywana temperatura
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Rzeczywista temperatura")
plt.ylabel("Przewidywana temperatura")
plt.title("Random Forest Regression: temperatura rzeczywista vs przewidywana")

min_temp = min(y_test.min(), y_pred.min())
max_temp = max(y_test.max(), y_pred.max())
plt.plot([min_temp, max_temp], [min_temp, max_temp])

plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 9. Ważność cech
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feature_importance)

plt.figure(figsize=(8, 5))
plt.bar(feature_importance["feature"], feature_importance["importance"])
plt.xlabel("Cecha")
plt.ylabel("Ważność")
plt.title("Ważność cech w modelu Random Forest")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
