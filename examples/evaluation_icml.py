import os.path
import numpy as np
import pandas as pd
import jpt.variables
import jpt.trees
import sklearn.model_selection
import time
import plotly.graph_objects as go

np.random.seed(69)
dataset_root = os.path.join("..", "..", "Documents", "datasets")
msls = list(reversed([0.01, 0.05, 0.1, 0.2, 0.4, 0.9]))
mvis = [0.1, 0.2, 0.5, 0.8, 1, 1.5]
print(mvis)
table = []


def iris():
    """https://archive.ics.uci.edu/ml/datasets/Iris"""

    names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"]
    data = pd.read_csv(os.path.join(dataset_root, "iris.data"), names=names)
    table.append(["IRIS Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished IRIS")


def adult():
    """https://archive.ics.uci.edu/ml/datasets/Adult"""
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data = pd.read_csv(os.path.join(dataset_root, "adult.data"), names=names)
    table.append(["Adult Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished adult")


def bean():
    """https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset"""
    data = pd.read_excel(os.path.join(dataset_root, "Dry_Bean_Dataset.xlsx"))
    table.append(["Dry Bean Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished bean")


def wine():
    """https://archive.ics.uci.edu/ml/datasets/Wine"""
    data = pd.read_csv(os.path.join(dataset_root, "wine.data"), names=["Alcohol", "Malic acid", "Ash",
                                                                       "Alcalinity of ash", "Magnesium",
                                                                       "Total phenols", "Flavanoids",
                                                                       "Nonflavanoid phenols",
                                                                       "Proanthocyanins", "Color intensity",
                                                                       "Hue", "OD280/OD315 of diluted wines",
                                                                       "Proline"])

    table.append(["Wine Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished wine")


def wine_quality():
    """https://archive.ics.uci.edu/ml/datasets/Wine+Quality"""
    red_data = pd.read_csv(os.path.join(dataset_root, "winequality-red.csv"), delimiter=";")
    red_data["color"] = "red"
    white_data = pd.read_csv(os.path.join(dataset_root, "winequality-white.csv"), delimiter=";")
    white_data["color"] = "white"
    data = pd.concat((red_data, white_data))

    table.append(["Wine Quality Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished wine quality")

def bank():
    """https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"""
    data = pd.read_csv(os.path.join(dataset_root, "bank-full.csv"), sep=";")
    table.append(["Bank and Marketing Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished bank")

def car():
    """https://archive.ics.uci.edu/ml/datasets/Car+Evaluation"""
    data = pd.read_csv(os.path.join(dataset_root, "car.data"), names=["buying", "maint", "doors", "persons", "lug_boot",
                                                                      "safety"])
    table.append(["Car evaluation Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    table.extend([list(a) for a in zip(mvis, params, trl, tel, zp)])
    print("finished car")


def raisin():
    """https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset"""
    data = pd.read_excel(os.path.join(dataset_root, "Raisin_Dataset.xlsx"))
    table.append(["Raisin Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished raisin")


def abalone():
    """https://archive.ics.uci.edu/ml/datasets/Abalone"""
    dataset = os.path.join(dataset_root, "abalone.data")
    data = pd.read_csv(dataset, names=["Sex", "Length", "Diameter", "Height", "Whole Height", "Shucked weight",
                                       "Viscera weight", "Shell weight", "Rings"])
    table.append(["Abalone Dataset", "Dataset size", len(data), "Number of variables", len(data.columns)])
    params, trl, tel, zp = run_experiments(data)
    print("finished abalone")

def evaluate_model(model, train, test):

    start = time.time()
    model.fit(train)
    model.postprocess_leaves()
    end = time.time()

    train_likelihood = model.likelihood(train)
    test_likelihood = model.likelihood(test)
    zero_likelihoods = test_likelihood[test_likelihood == 0]
    test_likelihood = test_likelihood[test_likelihood > 0]

    return end - start, model.number_of_parameters(), train_likelihood, test_likelihood, len(zero_likelihoods)/len(test)

def run_experiments(data):
    variables = jpt.variables.infer_from_dataframe(data, scale_numeric_types=False)
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.1)

    params = []
    trl = []
    tel = []
    zp = []

    for msl in msls:
        model = jpt.trees.JPT(variables, min_samples_leaf=msl)
        _, params_, trl_, tel_, zp_ = evaluate_model(model, train, test)
        params.append(params_)
        trl.append(np.sum(np.log(trl_)))
        tel.append(np.sum(np.log(tel_)))
        zp.append(zp_)
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])
    return params, trl, tel, zp


def plot_runs(parameters, train_likelihoods, test_likelihoods, zero_percentage):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=parameters, y=train_likelihoods, name="Train Performance"))
    fig.add_trace(go.Scatter(x=parameters, y=test_likelihoods, marker=dict(opacity=1 - zero_percentage),
                             name="Test Performance",
                  text=["Percentage of impossible samples: %s" % round(percentage * 100, 3)
                        for percentage in zero_percentage]))
    fig.update_layout(xaxis_title="Number of parameters",
                      yaxis_title="Log-Likelihood")
    return fig


def to_latex():
    df = pd.read_csv(os.path.join(dataset_root, "..", "jpt-results.csv"))
    print(df.to_latex(float_format="%.2f", index=False))


if __name__ == "__main__":
    to_latex()
    exit()
    iris()
    adult()
    bean()
    wine()
    wine_quality()
    bank()
    car()
    raisin()
    abalone()
    result = pd.DataFrame(table, columns=["Min samples per leaf", "Number of parameters", "Train Log-Likelihood",
                                          "Test Log-Likelihood", "%-age of 0 samples in test"])
    result.to_csv(os.path.join(dataset_root, "..", "jpt-results.csv"))

    print(result)