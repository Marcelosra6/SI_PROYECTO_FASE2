import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# =========================
# CARGAR DATASET
# =========================

df = pd.read_csv("dataset/sales_data_sample.csv", encoding="latin-1")

print("\nDATASET:")
print(df.head())

print("\nCOLUMNAS:")
print(df.columns.tolist())

# =========================
# LIMPIEZA
# =========================

target = "SALES"
feature_columns = ["QUANTITYORDERED", "PRICEEACH", "MSRP", "YEAR_ID"]

required_columns = feature_columns + [target]
df = df[required_columns].dropna()

print("\nREGISTROS UTILIZADOS:", len(df))

X = df[feature_columns]
y = df[target]

# =========================
# DIVIDIR DATOS
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODELOS
# =========================

models = {
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor())
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])
}

best_model = None
best_score = -999999
best_name = None

# =========================
# ENTRENAMIENTO
# =========================

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print(f"\nMEJOR MODELO: {best_name} (R2 = {best_score:.4f})")

# =========================
# GUARDAR MODELO
# =========================

joblib.dump(best_model, "model.pkl")
print("MODELO GUARDADO EN model.pkl")

# =========================
# GRAFICA
# =========================

predictions = best_model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:50], label="Real")
plt.plot(predictions[:50], label="Predicción")
plt.legend()
plt.title("Predicción vs Real")
plt.xlabel("Muestra")
plt.ylabel("Ventas")
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.close()
print("GRAFICA GUARDADA EN prediction_vs_actual.png")