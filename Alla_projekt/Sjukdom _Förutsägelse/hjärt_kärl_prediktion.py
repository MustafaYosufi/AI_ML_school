import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv("cardio_train.csv",sep = ";", index_col = "id").assign(
    age = lambda df: round(df["age"]/365),
    gender_cat = lambda df: df["gender"].map({2: "men", 1: "women"}))



columns = ["age", "gender", "height", "weight", "cholesterol", "smoke"]

fig, axes = plt.subplots(3, 2, figsize=(15, 7))
axes = axes.flatten()

for i, col in enumerate(columns):
    ax = axes[i]
    
    if col == "cholesterol":
        ax.pie(df[col].value_counts().values, labels=["Normal", "Above normal", "Far above normal"], autopct="%1.1f%%")
        ax.set_title("Cholesterol")
        
    elif col == "smoke":
        ax.pie(df[col].value_counts().values, labels=["Non-smoker", "Smoker"], autopct="%1.1f%%")
        ax.set_title("Smoking")
        
    elif col in ["age", "height", "weight"]:
        sns.histplot(data=df, x=col, bins=30, ax=ax)
        ax.set_title(f"{col.capitalize()} Distribution")
        
    elif col == "gender":
        sns.countplot(data = df, x="cardio", hue="gender_cat", ax=ax)
        ax.set_title("Cardio by Gender")

plt.tight_layout()
plt.show()