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
    tree = JPT(variables, min_samples_leaf=0.4)
    data = df.to_numpy()
    tree.fit(data.copy())
    print("finished learning")
    tree.plot(plotvars=variables,
              directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-abalone'))
    tree2 = tree.copy()
    tree2.postprocess_leaves()
    tree2.plot(plotvars=variables,
              directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-abalone2'))

    # print("finished plotting")

    l = tree.likelihood(data)
    print(sum(l==0))
    print(sum(np.log(l)))


if __name__ == "__main__":
    plants()
