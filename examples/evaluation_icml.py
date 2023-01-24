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

table = []


def iris():
    """https://archive.ics.uci.edu/ml/datasets/Iris"""

    names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"]
    data = pd.read_csv(os.path.join(dataset_root, "iris.data"), names=names)

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the IRIS dataset")
    # fig.show()
    table.append(["IRIS Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def adult():
    """https://archive.ics.uci.edu/ml/datasets/Adult"""
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data = pd.read_csv(os.path.join(dataset_root, "adult.data"), names=names)

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Adult dataset")
    #fig.show()
    table.append(["Adult Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def bean():
    """https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset"""
    data = pd.read_excel(os.path.join(dataset_root, "Dry_Bean_Dataset.xlsx"))

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Dry Bean dataset")
    # fig.show()
    table.append(["Dry Bean Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def wine():
    """https://archive.ics.uci.edu/ml/datasets/Wine"""
    data = pd.read_csv(os.path.join(dataset_root, "wine.data"), names=["Alcohol", "Malic acid", "Ash",
                                                                       "Alcalinity of ash", "Magnesium",
                                                                       "Total phenols", "Flavanoids",
                                                                       "Nonflavanoid phenols",
                                                                       "Proanthocyanins", "Color intensity",
                                                                       "Hue", "OD280/OD315 of diluted wines",
                                                                       "Proline"])

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Wine dataset")
    # fig.show()
    table.append(["Wine Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def wine_quality():
    """https://archive.ics.uci.edu/ml/datasets/Wine+Quality"""
    red_data = pd.read_csv(os.path.join(dataset_root, "winequality-red.csv"), delimiter=";")
    red_data["color"] = "red"
    white_data = pd.read_csv(os.path.join(dataset_root, "winequality-white.csv"), delimiter=";")
    white_data["color"] = "white"
    data = pd.concat((red_data, white_data))

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Wine Quality dataset")
    # fig.show()
    table.append(["Wine Quality Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def bank():
    """https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"""
    data = pd.read_csv(os.path.join(dataset_root, "bank-full.csv"), sep=";")
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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Bank Marketing dataset")
    # fig.show()
    table.append(["Bank and Marketing Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])

def car():
    """https://archive.ics.uci.edu/ml/datasets/Car+Evaluation"""
    data = pd.read_csv(os.path.join(dataset_root, "car.data"), names=["buying", "maint", "doors", "persons", "lug_boot",
                                                                      "safety"])
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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Car Evaluation dataset")
    # fig.show()
    table.append(["Car evaluation Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def raisin():
    """https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset"""
    data = pd.read_excel(os.path.join(dataset_root, "Raisin_Dataset.xlsx"))
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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Raisin dataset")
    # fig.show()

    table.append(["Raisin Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


def abalone():
    """https://archive.ics.uci.edu/ml/datasets/Abalone"""
    dataset = os.path.join(dataset_root, "abalone.data")
    data = pd.read_csv(dataset, names=["Sex", "Length", "Diameter", "Height", "Whole Height", "Shucked weight",
                                       "Viscera weight", "Shell weight", "Rings"])

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

    fig = plot_runs(params, trl, tel, np.array(zp))
    fig.update_layout(title="Results of experiments for the Abalone dataset")
    # fig.show()

    table.append(["Abalone Dataset", "Dataset size", len(data), "Number of variables", len(variables)])
    table.extend([list(a) for a in zip(msls, params, trl, tel, zp)])


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
    print(df.to_latex())


if __name__ == "__main__":
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