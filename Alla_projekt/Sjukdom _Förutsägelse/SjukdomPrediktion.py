import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import warnings

# === Ladda och förbehandla data ===
def load_data():
    # Läs in CSV-fil och sätt 'id' som index
    df = pd.read_csv("cardio_train.csv", sep=";", index_col="id")
    # Konvertera ålder från dagar till år
    df["age"] = (df["age"] / 365).round()
    # Koda kön som kategorier
    df["gender_cat"] = df["gender"].map({1: "women", 2: "men"})
    # Räkna ut BMI
    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
    return df

def clean_data(df):
    # Ta bort orimliga BMI-värden
    df = df.query("10 <= bmi <= 70").copy()
    # Skapa kategorier för BMI
    df["bmi_cat"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 35, 40, float('inf')],
                           labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Obese III"])
    # Justera blodtrycksvärden som verkar felaktiga
    df["ap_hi"] = df["ap_hi"].apply(lambda x: abs(x) * 10 if abs(x) < 50 else x / 10 if x > 340 else x)
    df["ap_lo"] = df["ap_lo"].apply(lambda x: x / 10 if x > 150 else x)
    # Filtrera bort orimliga blodtrycksmätningar
    df = df.query("ap_lo < ap_hi and ap_hi - ap_lo <= 185").copy()
    # Skapa blodtryckskategorier
    df["bl_tr_cat"] = pd.cut(df["ap_hi"], [0, 90, 120, 129, 139, 180, float("inf")],
                             labels=["Low", "Healthy", "Elevated", "Hypertension I", "Hypertension II", "Hypertension III"])
    return df

# Dela upp data i tränings-, validerings- och testmängder
def prepare_split(df, version=1):
    # Välj kolumner att ta bort beroende på version
    drop_cols = ["height", "weight", "ap_hi", "ap_lo", "bmi"] if version == 1 else ["height", "weight", "bmi_cat", "bl_tr_cat"]
    # Välj kategoriska kolumner att omvandla
    cat_cols = ["bmi_cat", "bl_tr_cat", "gender_cat"] if version == 1 else ["gender_cat"]
    # Skapa dummies (one-hot encoding) för kategoriska kolumner
    df = pd.get_dummies(df.drop(columns=drop_cols), columns=cat_cols, prefix="Col")
    # Dela upp i features och målvariabel
    X, y = df.drop("cardio", axis=1), df["cardio"]
    # Dela upp i tränings-, validerings- och testmängder
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test

# Visualisera enskilda variabler 
def visualize_individual(df):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cols = ["age", "gender", "height", "weight", "cholesterol", "smoke"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), cols):
        if col in ["cholesterol", "smoke"]:
            # Visa som cirkeldiagram
            labels = ["Normal", "Above", "Far above"] if col == "cholesterol" else ["Non-smoker", "Smoker"]
            ax.pie(df[col].value_counts(), labels=labels, autopct="%1.1f%%")
        elif col in ["age", "height", "weight"]:
            # Visa som histogram
            sns.histplot(df[col], bins=30, ax=ax)
        else:
            # Visa som stapeldiagram uppdelat efter kön
            sns.countplot(data=df, x="cardio", hue="gender_cat", ax=ax)
        ax.set_title(col.capitalize())
    
    plt.tight_layout()
    plt.savefig("indiviual_plot.png")
    plt.show()

# Visualisera kategoriska variabler 
def visualize_categories(df):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for ax, col in zip(axes, ["bl_tr_cat", "gender_cat", "bmi_cat"]):
        vals = df[df["cardio"] == 1][col].value_counts()
        sns.barplot(x=vals.index, y=vals.values, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig("categories_plot.png")
    plt.show()

# Visa korrelationer mellan numeriska variabler
def show_correlation(df):
    plt.figure(figsize=(10, 10))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, annot_kws={"size": 10}, cmap="coolwarm", fmt=".2f")
    plt.title("Korrelation mellan numeriska variabler")
    plt.savefig("korreklatin_plot.png")
    plt.tight_layout(); plt.show()
    
# Skapa modeller och deras hyperparametrar 
def create_models():
    return {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            "C": [0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear", "saga"]
        }),
        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "metric": ["euclidean", "minkowski"]
        }),
        "DecisionTree": (DecisionTreeClassifier(), {
            "max_depth": [5, 7, 9], "min_samples_split": [4, 6], "criterion": ["gini", "entropy"]
        }),
        "RandomForest": (RandomForestClassifier(), {
            "n_estimators": [80, 100], "max_depth": [10, 20], "min_samples_split": [2, 4]
        }),
        "XGBoost": (XGBClassifier(eval_metric="logloss"), {
            "n_estimators": [100], "learning_rate": [0.01, 0.1], "max_depth": [4, 5]
        })
    }

# Träna och utvärdera en enskild modell
def evaluate_model_by_name(name, df, version=1, verbose=False):
    models = create_models()
    
    if name not in models:
        print(f"Modellnamn '{name}' finns inte. Tillgängliga: {list(models.keys())}")
        return

    model, params = models[name]
    x_train, x_val, _, y_train, y_val, _ = prepare_split(df, version)

    # Skapa pipeline med standardisering, normalisering och modellen
    pipe = Pipeline([("scaler", StandardScaler()), ("normalization", MinMaxScaler()), ("model", model)])
    # Omvandla parameternamn för GridSearchCV
    param_grid = {f"model__{k}": v for k, v in params.items()}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(x_train, y_train)

    # Prediktion och utvärdering
    y_pred = grid.predict(x_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    accuracy = accuracy_score(y_val, y_pred)

    if verbose:
        result = {
            "model": name,
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "best_params": grid.best_params_
        }
        return grid.best_estimator_, result
    else: 
        return grid.best_estimator_

# Utvärdera en Voting Classifier som kombinerar flera modeller 
def evaluate_voting_classifier(df, version=1):
    model_names = ["LogisticRegression", "KNN", "XGBoost", "RandomForest", "DecisionTree"]
    estimators = []

    for name in model_names:
        best_model = evaluate_model_by_name(name, df, version, verbose=False)
        if best_model:
            estimators.append((name.lower(), best_model.named_steps["model"]))

    x_train, x_val, _, y_train, y_val, _ = prepare_split(df, version)

    voting_model = Pipeline([
        ("scaler", StandardScaler()),
        ("normalization", MinMaxScaler()), 
        ("model", VotingClassifier(estimators=estimators, voting="soft"))])

    voting_model.fit(x_train, y_train)
    y_pred = voting_model.predict(x_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    accuracy = accuracy_score(y_val, y_pred)

    result = {
        "model": "VotingClassifier",
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    return result

# Jämför olika modeller och visa resultatet 
def compare_models(df, version=1, verbose=False ):
    Voting_model = evaluate_voting_classifier(df, version)
    model_names = ["LogisticRegression", "KNN", "XGBoost", "RandomForest", "DecisionTree"]
    results = []

    for name in model_names:
        _, result = evaluate_model_by_name(name, df, version, verbose=True)
        if result:
            results.append(result)
        else:
            print(f"Modell {name} returnerade inget resultat!")
    
    results.append(Voting_model)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="accuracy", ascending=False)

    print("\nModelljämförelse:\n")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score"]].to_string(index=False))

    if verbose and "best_params" in results_df.columns:
        print("\nHyperparametrar:\n")
        print(results_df[["model", "best_params"]].dropna().to_string(index=False))

# Utvärdera XGBoost-modellen separat på testdatan
def evaluate_xgboost_classifier(df, version=2):
    _, _, x_test, _, _, y_test = prepare_split(df, version)
    model = evaluate_model_by_name("XGBoost", df, version=2, verbose=False)
    y_pred = model.predict(x_test)

    print("Klassificeringsrapport för XGBoost:\n", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Frisk (0)", "Sjuk (1)"], cmap="Blues")
    plt.title("Confusion Matrix för XGBoost")
    plt.show()

# Visualisera jämförelse mellan modeller i stapeldiagram 
def visualize_results():
    model_name = ["Random_forest", "DecisionTree", "XGBClassifier", "LogisticRegression", "Knn"]
    model_result = [[0.732, 0.67], [0.727, 0.69], [0.734, 0.69], [0.729, 0.64], [0.709, 0.69]]

    model_df = pd.DataFrame(model_result, columns=["Accuracy", "Recall"])
    model_df["Model"] = model_name
    df_melted = model_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted)
    plt.ylim(0.6, 0.75)  
    plt.title("Jämförelse av modeller: Accuracy och Recall")
    plt.xticks(rotation=30)  
    plt.savefig("result_plot.png")
    plt.show()