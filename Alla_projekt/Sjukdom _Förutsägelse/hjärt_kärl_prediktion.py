import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def loading_data():
    df = pd.read_csv("cardio_train.csv",sep = ";", index_col = "id").assign(
         age = lambda df: round(df["age"]/365),
         gender_cat = lambda df: df["gender"].map({2: "men", 1: "women"}),
         bmi = lambda df: round(df["weight"]/((df["height"]/100)*(df["height"]/100)), 2)) # Räknar ut BMI-värde
    return df

def visualizing_Individual_Data(df):

    columns = ["age", "gender", "height", "weight", "cholesterol", "smoke"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
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

def cleaning_fearture(df): 
    # Rensa orimliga BMI-värden
    df = df.query("10 <= bmi <= 70")

    # BMI-kategorier
    df["bmi_cat"] = pd.cut(df["bmi"],
                    bins=[0, 18.5, 25, 30, 35, 40, float('inf')],
                    labels=["Underweight", "Normal weight", "Overweight", "Obese I", "Obese II", "Obese III"])

    # Justera blodtryck
    # Negativa värden blir positiva, små värden skalas upp, extrema skalas ner
    df["ap_hi"] = df["ap_hi"].apply(
        lambda x: -x if x < 0 else x*10 if x < 50 else x/10 if 340 < x < 9999 else x/100 if x > 9999 else x)
    df["ap_lo"] = df["ap_lo"].apply(lambda x: x/10 if 150 < x < 9999 else x/100 if x > 9999 else x)

    #Ta bort orimliga blodtrycksvärden:
    # Ap_lo måste vara mindre än ap_hi och skillnaden mellan ap_hi och ap_lo får inte vara större än 185
    df = df.query("ap_lo < ap_hi and ap_hi - ap_lo <= 185")

    # Skapa blodtryckskategorier baserat på ap_hi
    df["bl_tr_cat"] = pd.cut(df["ap_hi"],
                        bins=[0, 90, 120, 129, 139, 180, float("inf")],
                        labels=["Low", "Healthy", "Elevated", "Hypertension I", "Hypertension II", "Hypertension III"])
    
    return df


df = cleaning_fearture(loading_data())

def visualizing_cat_data(df):
    data = df[df["cardio"] == 1]
    columns =["bl_tr_cat", "gender_cat", "bmi_cat"]
    fig, axes = plt.subplots(3, 1, figsize = (8, 6))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        col_data = data[col].value_counts().sort_values()
        sns.barplot(x = col_data.index, y = col_data.values, ax = axes[i])
        axes[i].set_title(col)
    
    plt.tight_layout()

    plt.show()

def find_correlation(df):
    df_number = df.select_dtypes(include ="number")
    df_corr = df_number.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(data = df_corr, annot = True)
    plt.show()

