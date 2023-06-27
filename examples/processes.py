import numpy as np
from matplotlib import pyplot as plt

from jpt import NumericVariable, JPT
from jpt.distributions import Numeric
from jpt.sequential_trees import SequentialJPT
from jpt.variables import VariableMap


def sample(n):
    for x in np.linspace(0, np.pi * 4, 100):
        yield np.sin(x) + np.random.uniform(-.1, .1)


def main():
    data = []
    for _ in range(10):
        yy = list(sample(100))
        plt.plot(list(range(len(yy))), yy)
        data.append(np.array(yy).reshape([-1, 1]))
    # plt.show()
    # print(data)
    y = NumericVariable('y', domain=Numeric)
    trans = JPT([y], min_samples_leaf=.2)
    seq = SequentialJPT(trans)
    seq.fit(data)

    # for v in seq.template_tree.variables:
    #     print(v)
    # print(seq.transition_model)
    print(seq.template_tree.plot(plotvars=['y'], view=True))

    for result in seq.posterior([VariableMap()] * 100):
        # print(result.posterior([y], evidence={}).distributions[y].cdf)
        result.posterior([y], evidence={}).distributions[y].plot(view=True)



if __name__ == '__main__':
    main()
