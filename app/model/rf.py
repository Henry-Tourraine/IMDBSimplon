from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model.dataframe import IMDB_df
import pickle
import os

def save_model(df: IMDB_df):
    param_grid = {"n_estimators": [200], "min_samples_split": [3], "min_samples_leaf": [2]}

    corr = df.top_correlated("imdb_score", n=30)

    X, y = df[[c for c in df.columns if c != "imdb_score" and c in corr]], df["imdb_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=200, min_samples_split=3, min_samples_leaf=2)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # le scale n'a pas d'impact sur la prédiction
    predictions = model.predict(X_test)


    mse = mean_squared_error(y_test, predictions)
    print(f"columns, mse: {mse}, rf score : {model.score(X_test, y_test)}")
    pickle.dump(model, open(os.environ["MODEL"], "wb"))
    return model


def get_model(df: IMDB_df)->RandomForestClassifier:
    if not os.path.exists(os.environ["MODEL"]):
        return save_model(df)
    return pickle.load(open(os.environ["MODEL"], "rb"))