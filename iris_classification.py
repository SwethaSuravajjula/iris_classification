import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_iris_dataframe():
    """
    Load the Iris dataset from scikit-learn and return a pandas DataFrame
    containing the features and a 'species' column.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.target, iris.target_names


def exploratory_data_analysis(df):
    """
    Perform basic EDA: print head and summary statistics, then plot a pairplot.
    """
    print("First five rows of the Iris DataFrame:")
    print(df.head(), "\n")

    print("Summary statistics:")
    print(df.describe(), "\n")

    # Pairplot colored by species
    sns.set(style="ticks")
    pairplot = sns.pairplot(df, hue='species', diag_kind='hist')
    pairplot.fig.suptitle("Iris Pairplot (features vs. species)", y=1.02)
    plt.show()


def train_random_forest(X_train, y_train):
    """
    Initialize and train a Random Forest classifier on the training data.
    Returns the trained model.
    """
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test, target_names):
    """
    Use the trained model to predict on X_test, then print accuracy and a full
    classification report.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


def main():
    # 1. Load the data into a DataFrame
    df, y, target_names = load_iris_dataframe()

    # 2. (Optional) Exploratory Data Analysis
    exploratory_data_analysis(df)

    # 3. Prepare features and labels for modeling
    X = df.drop(columns=['species'])
    y = df['species'].apply(lambda name: list(target_names).index(name)).values

    # 4. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5. Train a Random Forest classifier
    model = train_random_forest(X_train, y_train)

    # 6. Evaluate on the test set
    evaluate_model(model, X_test, y_test, target_names)

    # 7. (Optional) Feature importances bar plot
    importances = model.feature_importances_
    feature_names = X.columns
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
