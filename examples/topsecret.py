import pandas
import os
import pickle
from jpt.variables import infer_from_dataframe
import jpt.trees
import numpy as np
import numpy.lib.stride_tricks
from datetime import datetime

def main():
    path = os.path.join("..", "..", "Downloads", "travel-data")

    with open(os.path.join(path, "Basket 1 (AYT) - Low Season 2018-br.df"), "rb") as file:
        df = pickle.load(file)

    variables = infer_from_dataframe(df)

    tree = jpt.trees.JPT(variables, min_samples_leaf=0.05)
    tree.fit(df)
    #tree.plot(plotvars=tree.variables,
     #         directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-travel-data'), view=False)

if __name__ == "__main__":
    main()
