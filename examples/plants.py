import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import dnutils
from jpt.trees import JPT
from jpt.variables import infer_from_dataframe, VariableMap


def plants():
    path = os.path.join("..", "examples", "data", "abalone.data")
    df = pd.read_csv(path, names=["Sex", "Length", "Diameter", "Height", "Whole weight",
                                  "Shucked weight", "Viscera weight", "Shell weight", "Rings"])

    variables = infer_from_dataframe(df, scale_numeric_types=False)
    print(variables)
    tree = JPT(variables, min_samples_leaf=0.02)
    data = df.to_numpy()
    tree.fit(data.copy())
    tree.postprocess_leaves()
    print("finished learning")
    #tree.plot(plotvars=variables,
     #         directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-abalone'))

    print("finished plotting")

    l = tree.likelihood(data)
    l[l==0] = pow(1/len(data), len(variables))
    print(sum(np.log(l)))


if __name__ == "__main__":
    plants()
