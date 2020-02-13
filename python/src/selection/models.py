from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump


def train_model_gridsearch(model, X, y, cv, parameters, output_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                        random_state=42)

    grid_search = GridSearchCV(model, parameters, cv=cv)
    grid_search.fit(X_train, y_train)
    grid_search.score(X_test, y_test)

    print(f"Generalization score for polynomial kernel: train:"
          f"{grid_search.score(X_train, y_train)}, test: {grid_search.score(X_test, y_test)}")
    print('Best params: ', grid_search.best_params_)

    # Save the model
    if output_path is not None:
        dump(grid_search, output_path)

    return grid_search
