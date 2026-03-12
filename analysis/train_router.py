import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("datasets/router_dataset.csv")

X = df[["edge_density", "color_variance", "high_freq_energy"]]
y = df["best_model"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))

print("feature importance:", model.feature_importances_)