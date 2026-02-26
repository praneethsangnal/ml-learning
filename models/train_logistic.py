import pandas as pd
from preprocessing.clean_titanic import clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def train_model():

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv("datasets/titanic.csv")

    # ----------------------------
    # Clean Data
    # ----------------------------
    df = clean_data(df)

    # ----------------------------
    # Separate Features & Target
    # ----------------------------
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # ----------------------------
    # Train-Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # adding feature scaling 
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    # ----------------------------
    # Train Logistic Regression
    # ----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred = model.predict(X_test)

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("first five row\n",df.head())


if __name__ == "__main__":
    train_model()