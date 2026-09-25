import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import os
    import random

    import torch
    import optuna
    import numpy as np
    import seaborn as sns
    import polars as pl
    import matplotlib.pyplot as plt

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer

    from tqdm import tqdm
    from pathlib import Path

    from utils import set_seed

    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        Path,
        Ridge,
        StratifiedKFold,
        accuracy_score,
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
    set_seed(0xFACED)


    OPTUNA_SEARCH = True
    return (OPTUNA_SEARCH,)


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
def _(train_dataset):
    X, y = train_dataset.drop_nulls(['Embarked']).drop(['Survived']), train_dataset.drop_nulls(['Embarked'])['Survived']
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Убираем ['Name', 'Ticket', 'Cabin']
    """)
    return


@app.cell
def _(pl):
    def prepare_dataset(df: pl.DataFrame):
        # убираем ненужные фичи
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'])

        # заполняем пропуски в age
        df = df.fill_null(df['Age'].median())

        # дропаем nulls в embarked, т.к. там всего их 2 и не имеет смысла закидывать в onehot, вообще они дропаются сверху
        df = df.drop_nulls(['Embarked'])

        # переводим в onehot категориальные
        df = df.to_dummies(['Sex', 'Embarked'])

        return df

    return (prepare_dataset,)


@app.cell
def _(X, prepare_dataset, test_dataset, train_test_split, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    X_train, X_val, X_test = map(prepare_dataset, (X_train, X_val, test_dataset))
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## линейная регрессия
    """)
    return


@app.cell
def _(ElasticNet, Lasso, LinearRegression, Ridge, StratifiedKFold, optuna):
    params_lin = {
        'LinearRegression': {},
        'Lasso': {
            'alpha': 1,
        },
        'Ridge': {
            'alpha': 100.0,
        },
        'ElasticNet': {
            'alpha': 1.0,
            'l1_ratio': 0.8,
        },
        'StratifiedKFold': {
            'n_splits': 5,
        }
    }

    param_distributions = {
        "LinearRegression": {},

        "Lasso": {
            "alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "Ridge": {
            "alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "ElasticNet": {
            "alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
            "l1_ratio": optuna.distributions.FloatDistribution(
                0.0, 1.0
            ),
        },
    }

    models_lin = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet()
    }


    cv = StratifiedKFold(**params_lin['StratifiedKFold'])
    return cv, models_lin, param_distributions, params_lin


@app.cell
def _(
    X_train,
    X_val,
    accuracy_score,
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


    def start_fits(models, cv, X_train, y_train, X_val, y_val):
        results = []

        for model_name, model in models.items():
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


    results = start_fits(
        models_lin,
        cv,
        X_train,
        y_train,
        X_val,
        y_val
    )

    results
    return (scorer_accuracy,)


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    cv,
    make_scorer,
    models_lin,
    optuna,
    param_distributions,
    params_lin,
    scorer_accuracy,
    y_train,
):
    if OPTUNA_SEARCH:
        searches = {}

        for name, model in models_lin.items():
            search = optuna.integration.OptunaSearchCV(
                estimator=model,
                param_distributions=param_distributions[name],
                cv=cv,
                verbose=0,
                scoring=make_scorer(scorer_accuracy),
                n_trials=100 if param_distributions[name] else 1,
            )
        
            search.fit(X_train, y_train)


            searches['name'] = search

            for param_name, param_value in search.best_params_.items():
                params_lin[name][param_name] = param_value

            print(name)
            print("score:", search.best_score_)
            print("params:", search.best_params_)
            print()
    return


if __name__ == "__main__":
    app.run()
