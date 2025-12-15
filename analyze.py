import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset/clean_dataset.csv")
df = df[df["class"] == "Economy"].reset_index(drop=True)
# price is already numeric
df["price"] = df["price"].astype(float)

mean_price = df.groupby("days_left")["price"].mean()

plt.figure(figsize=(8,5))
plt.plot(mean_price.index, mean_price.values)
plt.xlabel("Days Left")
plt.ylabel("Average Price")
plt.title("Average Price vs Days Left")
plt.tight_layout()
plt.savefig("average_price_vs_days_left_economy.png")


plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=50)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Economy Class Flight Prices")
plt.tight_layout()
plt.savefig('price_distribution_economy.png')

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(df["price"]), bins=50)
plt.xlabel("log(Price)")
plt.ylabel("Frequency")
plt.title("Log-Transformed Price Distribution (Economy Class)")
plt.tight_layout()
plt.savefig('log_price_distribution_economy.png')

df_all = pd.read_csv("./dataset/clean_dataset.csv")

plt.figure(figsize=(8,5))
plt.hist(df_all[df_all["class"]=="Economy"]["price"], bins=50, alpha=0.7, label="Economy")
plt.hist(df_all[df_all["class"]=="Business"]["price"], bins=50, alpha=0.7, label="Business")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.title("Price Distribution by Class")
plt.tight_layout()
plt.show()

