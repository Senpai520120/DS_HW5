# ============================================================
#  Аналіз набору даних Iris
#  Інструменти: pandas, scikit-learn, matplotlib, seaborn
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # без GUI — зберігаємо у файл
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# КРОК 0 — Створення iris.csv
# ─────────────────────────────────────────────
print("=" * 60)
print("КРОК 0: Створення iris.csv")
print("=" * 60)

iris_raw = load_iris()
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

df_create = pd.DataFrame(iris_raw.data,
                         columns=["SepalLengthCm", "SepalWidthCm",
                                  "PetalLengthCm", "PetalWidthCm"])
df_create["Species"] = [species_map[t] for t in iris_raw.target]
df_create.to_csv("iris.csv", index=False)

print("Файл iris.csv успішно створено!")
print("\nПерші рядки файлу:")
print(df_create.head())


# ─────────────────────────────────────────────
# ЗАВДАННЯ 1 — Завантаження CSV та базова інформація
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 1: Завантаження iris.csv")
print("=" * 60)

df = pd.read_csv("iris.csv")

print(f"\nФорма даних (shape): {df.shape}")
print("\nТипи даних (dtypes):")
print(df.dtypes)
print("\nПерші 3 рядки:")
print(df.head(3))


# ─────────────────────────────────────────────
# ЗАВДАННЯ 2 — Інформація через sklearn load_iris
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 2: Об'єкт load_iris() зі sklearn")
print("=" * 60)

iris = load_iris()

print(f"\nКлючі об'єкта: {list(iris.keys())}")
print(f"Кількість рядків: {iris.data.shape[0]}")
print(f"Кількість стовпців: {iris.data.shape[1]}")
print(f"\nНазви ознак (feature_names):\n{list(iris.feature_names)}")
print(f"\nОпис датасету (DESCR — перші 1500 символів):")
print(iris.DESCR[:1500])


# ─────────────────────────────────────────────
# ЗАВДАННЯ 3 — Описова статистика
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 3: Описова статистика (describe)")
print("=" * 60)

print("\nСтатистика числових ознак:")
print(df.describe())


# ─────────────────────────────────────────────
# ЗАВДАННЯ 4 — Спостереження по видах
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 4: Дані згруповані за видом")
print("=" * 60)

for species, group in df.groupby("Species"):
    print(f"\n--- {species} ({len(group)} записів) ---")
    print(group.to_string(index=False))


# ─────────────────────────────────────────────
# ЗАВДАННЯ 5 — Boxplot по всіх ознаках
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 5: Boxplot — загальна статистика ознак")
print("=" * 60)

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Boxplot ознак датасету Iris за видами", fontsize=14)

for ax, col in zip(axes.flatten(), feature_cols):
    df.boxplot(column=col, by="Species", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Вид")
    ax.set_ylabel("Значення (см)")

plt.tight_layout()
plt.savefig("iris_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("Графік збережено у iris_boxplot.png")


# ─────────────────────────────────────────────
# ЗАВДАННЯ 6 — Bar chart частоти видів
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 6: Bar chart — частота видів Iris")
print("=" * 60)

species_counts = df["Species"].value_counts()
print(f"\nКількість записів за видами:\n{species_counts}")

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#4C72B0", "#DD8452", "#55A868"]
species_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
ax.set_title("Частота видів Iris", fontsize=13)
ax.set_xlabel("Вид")
ax.set_ylabel("Кількість записів")
ax.set_xticklabels(species_counts.index, rotation=20, ha="right")

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            int(bar.get_height()),
            ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("iris_species_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("Діаграму збережено у iris_species_bar.png")


# ─────────────────────────────────────────────
# ЗАВДАННЯ 7 — Поділ на X (ознаки) та y (мітки)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 7: Розділення на X (ознаки) та y (мітки)")
print("=" * 60)

X = df[feature_cols]
y = df["Species"]

print(f"\nX — ознаки, форма: {X.shape}")
print(X.head(3))
print(f"\ny — мітки, форма: {y.shape}")
print(y.head(3))


# ─────────────────────────────────────────────
# ЗАВДАННЯ 8 — train_test_split 70% / 30%
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 8: Розділення 70% / 30% (105 / 45 записів)")
print("=" * 60)

X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X, y, test_size=0.30, random_state=42)

print(f"\nТренувальний набір X (70%): {X_train_70.shape}")
print(X_train_70.head(5))
print(f"\nТренувальний набір y (70%): {y_train_70.shape}")
print(y_train_70.head(5).to_string())

print(f"\nТестовий набір X (30%): {X_test_30.shape}")
print(X_test_30.head(5))
print(f"\nТестовий набір y (30%): {y_test_30.shape}")
print(y_test_30.head(5).to_string())


# ─────────────────────────────────────────────
# ЗАВДАННЯ 9 — Кодування міток + розділення 80/20
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 9: Кодування Species + розділення 80% / 20% (120 / 30)")
print("=" * 60)

# Кодуємо текстовий вид у числовий
df_encoded = df.copy()
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df_encoded["Species"] = df_encoded["Species"].map(label_map)

print("\nДані після кодування (перші 5 рядків):")
print(df_encoded.head())

X_enc = df_encoded[feature_cols]
y_enc = df_encoded["Species"]

X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X_enc, y_enc, test_size=0.20, random_state=42)

print(f"\nТренувальний набір X (80%): {X_train_80.shape}")
print(X_train_80.head(5))
print(f"\nТренувальний набір y (80%): {y_train_80.shape}")
print(y_train_80.head(5).to_string())

print(f"\nТестовий набір X (20%): {X_test_20.shape}")
print(X_test_20.head(5))
print(f"\nТестовий набір y (20%): {y_test_20.shape}")
print(y_test_20.head(5).to_string())


# ─────────────────────────────────────────────
# ЗАВДАННЯ 10 — KNN класифікатор (70/30)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАВДАННЯ 10: KNeighborsClassifier (n_neighbors=5), розділення 70/30")
print("=" * 60)

# Використовуємо закодовані числові мітки для KNN
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(
    X_enc, y_enc, test_size=0.30, random_state=42)

print(f"\nТренувальний набір: {X_knn_train.shape[0]} записів")
print(f"Тестовий набір:     {X_knn_test.shape[0]} записів")

# Навчання моделі
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_knn_train, y_knn_train)

# Прогнозування
y_pred = knn.predict(X_knn_test)
accuracy = accuracy_score(y_knn_test, y_pred)

# Зворотна карта для читабельності
inv_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

print("\nПрогнози KNN для тестового набору (перші 20):")
results = X_knn_test.copy()
results["Реальний клас"]    = y_knn_test.map(inv_map).values
results["Прогноз KNN"]      = pd.Series(y_pred, index=X_knn_test.index).map(inv_map)
results["Правильно?"]       = results["Реальний клас"] == results["Прогноз KNN"]
print(results.head(20).to_string())

print(f"\nТочність (accuracy) моделі KNN: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "=" * 60)
print("Аналіз завершено! Збережені файли:")
print("  iris.csv           — датасет")
print("  iris_boxplot.png   — boxplot ознак за видами")
print("  iris_species_bar.png — стовпчаста діаграма видів")
print("=" * 60)
