import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("datasets/router_dataset.csv")

T = 2000

pred = df["high_freq_energy"].apply(
    lambda x: "swinir" if x < T else "realesrgan"
)

acc = (pred == df["best_model"]).mean()

print("threshold value:", T)
print("accuracy:", acc)
print(classification_report(df["best_model"], pred))