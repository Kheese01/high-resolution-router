import pandas as pd

df = pd.read_csv("datasets/router_dataset.csv")

print(df.head())

print(df["best_model"].value_counts())

print(df.describe())

print(df.groupby("best_model").mean())

print(f"Max value of high_freq_energy using SWINIR model: ", df[df["best_model"] == "swinir"]["high_freq_energy"].max())