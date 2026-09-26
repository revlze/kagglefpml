import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import os
    import random

    # Обход падения OpenMP на macOS; задаём до импорта ML-библиотек.
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    import optuna
    import numpy as np
    import seaborn as sns
    import polars as pl
    import matplotlib.pyplot as plt

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from tqdm import tqdm
    from pathlib import Path

    from utils import set_seed

    return (
        ColumnTransformer,
        DecisionTreeClassifier,
        ElasticNet,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        LGBMClassifier,
        Lasso,
        LinearRegression,
        OneHotEncoder,
        Path,
        Pipeline,
        RandomForestClassifier,
        Ridge,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        XGBClassifier,
        accuracy_score,
        clone,
        f1_score,
        make_scorer,
        np,
        optuna,
        pl,
        set_seed,
        tqdm,
        train_test_split,
    )


@app.cell
def _(set_seed):
    SEED = 0xFACED

    set_seed(SEED)


    OPTUNA_SEARCH = True
    return OPTUNA_SEARCH, SEED


@app.cell
def _(Path):
    base_path = Path('/users/platon/Documents/final_proj/')
    data_path = base_path / 'data/titanic/'
    return (data_path,)


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / 'train.csv')
    test_dataset = pl.read_csv(data_path / 'test.csv')
    return test_dataset, train_dataset


@app.cell
def _(test_dataset, train_dataset):
    # убираем ненужные фичи
    train_set= train_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_set = test_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'])
    return test_set, train_set


@app.cell
def _(train_set):
    X, y = train_set.drop('Survived'), train_set['Survived']
    return X, y


@app.cell
def _(X, test_set, train_test_split, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    X_test = test_set
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## preprocess
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
):
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ('numeric', numeric, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('categorical', categorical, ['Sex', 'Embarked']),
    ])
    return (preprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## линейная регрессия
    """)
    return


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    Pipeline,
    Ridge,
    StratifiedKFold,
    optuna,
    preprocess,
):
    param_distributions_lin = {
        "LinearRegression": {},

        "Lasso": {
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "Ridge": {
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "ElasticNet": {
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
            "model__l1_ratio": optuna.distributions.FloatDistribution(
                0.0, 1.0
            ),
        },
    }

    models_lin = {
        'LinearRegression': Pipeline([
            ('preprocess', preprocess),
            ('model', LinearRegression()),
        ]),
        'Lasso': Pipeline([
            ('preprocess', preprocess),
            ('model', Lasso()),
        ]),
        'Ridge': Pipeline([
            ('preprocess', preprocess),
            ('model', Ridge()),
        ]),
        'ElasticNet': Pipeline([
            ('preprocess', preprocess),
            ('model', ElasticNet()),
        ])
    }


    cv = StratifiedKFold(n_splits=5)
    return cv, models_lin, param_distributions_lin


@app.cell
def _(
    X_train,
    X_val,
    accuracy_score,
    clone,
    cv,
    f1_score,
    models_lin,
    np,
    pl,
    tqdm,
    y_train,
    y_val,
):
    def scorer_accuracy(y_true, y_pred):
            y_pred = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred)

    def scorer_f1(y_true, y_pred):
            y_pred = (y_pred > 0.5).astype(int)
            return f1_score(y_true, y_pred)

    def fit_single_model(model_name, model, cv, X_train, y_train):
        print(f"Started cv for model: {model_name}")

        scores = {
            "model": model_name,
            "train_acc": [],
            "train_f1": [],
            "cv_acc": [],
            "cv_f1": [],
        }

        for train_i, val_i in tqdm(
                                    cv.split(X_train, y_train),
                                    total=cv.get_n_splits(),
                                    leave=False
                                ):
            X_train_fold, y_train_fold = X_train[train_i], y_train[train_i]
            X_val_fold, y_val_fold = X_train[val_i], y_train[val_i]

            model.fit(X_train_fold, y_train_fold)

            pred_train = model.predict(X_train_fold)
            pred_val = model.predict(X_val_fold)

            scores["train_acc"].append(
                scorer_accuracy(y_train_fold, pred_train)
            )
            scores["train_f1"].append(
                scorer_f1(y_train_fold, pred_train)
            )

            scores["cv_acc"].append(
                scorer_accuracy(y_val_fold, pred_val)
            )
            scores["cv_f1"].append(
                scorer_f1(y_val_fold, pred_val)
            )

        result = {
            "model": scores["model"],

            "train_acc": np.round(np.mean(scores["train_acc"]), 4),
            "train_f1": np.round(np.mean(scores["train_f1"]), 4),

            "cv_acc": np.round(np.mean(scores["cv_acc"]), 4),
            "cv_f1": np.round(np.mean(scores["cv_f1"]), 4),
        }

        print(f"Ended cv for model: {model_name}")

        return result


    def start_fits(models, params, cv, X_train, y_train, X_val, y_val):
        results = []

        for model_name, model in models.items():
            model = clone(model)
            if params is not None:
                model.set_params(**params[model_name])
            result = fit_single_model(
                model_name,
                model,
                cv,
                X_train,
                y_train
            )

            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)

            result["val_acc"] = np.round(scorer_accuracy(y_val, pred_val), 4)
            result["val_f1"] = np.round(scorer_f1(y_val, pred_val), 4)

            results.append(result)

        return pl.DataFrame(results)


    results_lin = start_fits(
        models_lin,
        None,
        cv,
        X_train,
        y_train,
        X_val,
        y_val
    )

    results_lin
    return scorer_accuracy, start_fits


@app.cell
def _(SEED, optuna):
    def start_optuna(models, param_distributions, cv, scoring, X_train, y_train, random_state=SEED, n_trials=100, n_jobs=-1):
        searches = {}
        optuna_params = {name: {} for name in models}
        for name, model in models.items():
            print(f"Started Optuna for model: {name}", flush=True)
            search = optuna.integration.OptunaSearchCV(
                estimator=model,
                param_distributions=param_distributions[name],
                cv=cv,
                verbose=0,
                scoring=scoring,
                n_trials=n_trials if param_distributions[name] else 1,
                random_state=random_state,
                n_jobs=n_jobs
            )

            search.fit(X_train, y_train)
            print(f"Ended Optuna for model: {name}", flush=True)

            searches[name] = search

            for param_name, param_value in search.best_params_.items():
                optuna_params[name][param_name] = param_value

        return optuna_params


    return (start_optuna,)


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    make_scorer,
    models_lin,
    param_distributions_lin,
    scorer_accuracy,
    start_fits,
    start_optuna,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_params_lin = start_optuna(
            models_lin, param_distributions_lin, cv, make_scorer(scorer_accuracy),
            X_train, y_train
        )
        optuna_results_lin = start_fits(
        models_lin,
        optuna_params_lin,
        cv,
        X_train, y_train, X_val, y_val
    )

    optuna_results_lin
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KNN
    """)
    return


@app.cell
def _(KNeighborsClassifier, Pipeline, optuna, preprocess):
    models_knn = {
        'KNeighborsClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', KNeighborsClassifier()),
        ]),
    }
    # n_neighbors, weight, metric
    param_distributions_knn = {
        'KNeighborsClassifier': {
            'model__n_neighbors': optuna.distributions.IntDistribution(1, 100),
            'model__weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
            'model__metric': optuna.distributions.CategoricalDistribution(
                [
                    'cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                    'manhattan', 'nan_euclidean', 'minkowski'
                ]
            )
        },
    }
    return models_knn, param_distributions_knn


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    make_scorer,
    models_knn,
    param_distributions_knn,
    scorer_accuracy,
    start_fits,
    start_optuna,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_params_knn = start_optuna(
            models_knn, param_distributions_knn, cv, make_scorer(scorer_accuracy),
            X_train, y_train
        )
        optuna_results_knn = start_fits(
            models_knn, optuna_params_knn, cv,
            X_train, y_train, X_val, y_val)

    optuna_results_knn
    return


@app.cell
def _(X_train, X_val, cv, models_knn, start_fits, y_train, y_val):
    default_result_knn = start_fits(models_knn, None, cv, X_train, y_train, X_val, y_val)
    default_result_knn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DecisionTreeClassifier
    """)
    return


@app.cell
def _(DecisionTreeClassifier, Pipeline, optuna, preprocess):
    model_tree = {
        'DecisionTreeClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', DecisionTreeClassifier()),
        ])
    }

    param_distributions_tree = {
        'DecisionTreeClassifier': {
            'model__criterion': optuna.distributions.CategoricalDistribution(['gini', 'entropy', 'log_loss']),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 10),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 10)
        }
    }
    return model_tree, param_distributions_tree


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    make_scorer,
    model_tree,
    param_distributions_tree,
    scorer_accuracy,
    start_fits,
    start_optuna,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_params_tree = start_optuna(
            model_tree,
            param_distributions_tree,
            cv,
            make_scorer(scorer_accuracy),
            X_train, y_train
        )
        optuna_results_tree = start_fits(
            model_tree, optuna_params_tree,
            cv, X_train, y_train, X_val, y_val
        )
    optuna_results_tree, optuna_params_tree
    return


@app.cell
def _(X_train, X_val, cv, model_tree, start_fits, y_train, y_val):
    start_fits(model_tree, None, cv,
                X_train, y_train, X_val, y_val)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    видно, что обычное дерево переобучилось
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RandomForest
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, optuna, preprocess):
    model_rf = {
        'RandomForestClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', RandomForestClassifier(n_jobs=-1))
        ])
    }

    param_distributions_rf = {
        'RandomForestClassifier': {
            'model__n_estimators': optuna.distributions.IntDistribution(10, 200),
            'model__max_depth': optuna.distributions.IntDistribution(2, 60),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 20),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
        }
    }
    return model_rf, param_distributions_rf


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    make_scorer,
    model_rf,
    param_distributions_rf,
    scorer_accuracy,
    start_fits,
    start_optuna,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_params_rf = start_optuna(
            model_rf, param_distributions_rf,
            cv, make_scorer(scorer_accuracy),
            X_train, y_train
        )
        optuna_results_rf = start_fits(
            model_rf,
            optuna_params_rf,
            cv,
            X_train, y_train, X_val, y_val
        )
    optuna_results_rf, optuna_params_rf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boostings
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    LGBMClassifier,
    Pipeline,
    XGBClassifier,
    optuna,
    preprocess,
):
    model_gb = {
        'GradientBoostingClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', GradientBoostingClassifier()),
        ]),
        'LGBMClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', LGBMClassifier(n_jobs=1, verbosity=-1)),
        ]),
        'XGBClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', XGBClassifier(n_jobs=1)),
        ]),
    }

    param_distributions_gb = {
        'GradientBoostingClassifier': {
            'model__loss': optuna.distributions.CategoricalDistribution(['log_loss', 'exponential']),
            'model__learning_rate': optuna.distributions.FloatDistribution(1e-4, 1e-1),
            'model__n_estimators': optuna.distributions.IntDistribution(10, 200),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 50),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 50),
            'model__max_depth': optuna.distributions.IntDistribution(1, 10)
        },
        'LGBMClassifier': {
            'model__num_leaves': optuna.distributions.IntDistribution(3, 100),
            'model__max_depth': optuna.distributions.IntDistribution(1, 20),
            'model__learning_rate': optuna.distributions.FloatDistribution(1e-4, 1e-1),
            'model__n_estimators': optuna.distributions.IntDistribution(100, 1000),
            'model__reg_alpha': optuna.distributions.FloatDistribution(1e-3, 1),
            'model__reg_lambda': optuna.distributions.FloatDistribution(1e-3, 1),
        },
        'XGBClassifier': {
            'model__eta': optuna.distributions.FloatDistribution(1e-4, 1),
            'model__gamma': optuna.distributions.IntDistribution(0, 100),
            'model__max_depth': optuna.distributions.IntDistribution(1, 32),
        }
    }
    return model_gb, param_distributions_gb


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    make_scorer,
    model_gb,
    param_distributions_gb,
    scorer_accuracy,
    start_fits,
    start_optuna,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_params_gb = start_optuna(
            model_gb, param_distributions_gb,
            cv, make_scorer(scorer_accuracy),
            X_train, y_train
        )
        optuna_results_gb = start_fits(
            model_gb,
            optuna_params_gb,
            cv,
            X_train, y_train, X_val, y_val
        )
    optuna_results_gb, optuna_params_gb
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
