import pandas
import os
import pickle
from jpt.variables import infer_from_dataframe
import jpt.trees
import sequential_trees
import numpy as np
from datetime import datetime
import plotly.express as px


def main():
    start = datetime.now()

    path = os.path.join("..", "..", "Downloads", "travel-data")

    dfs = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        df_ = pandas.read_pickle(os.path.join(path, filename))
        if len(df_) > 0 and len(df_.columns) == 9:
            del df_["event_date"]
            dfs.append(df_)

    variables = infer_from_dataframe(pandas.concat(dfs), scale_numeric_types=False)

    targets = [variable for variable in variables if variable.name == "avg_pppnc_hotel_touroperator"]

    template_tree = jpt.trees.JPT(variables, min_samples_leaf=0.2)

    stree = sequential_trees.SequentialJPT(template_tree)

    stree.fit([df.to_numpy() for df in dfs])
    evidence_t0 = dict((column, value) for column, value in zip(dfs[-1].columns, dfs[-1].iloc[-1]))

    evidence = [evidence_t0] + [{}] * 10

    # stree.template_tree.plot(directory=os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-travel'))

    ims = stree.independent_marginals(evidence)

    for idx, tree in enumerate(ims[1:]):
        tree.plot(plotvars=tree.variables)
        e = tree.expectation(targets, evidence={}).result
        print(e)
    exit()
    fig = px.line(dfs[-1]["avg_pppnc_hotel_touroperator"])
    exit()






if __name__ == "__main__":
    main()
