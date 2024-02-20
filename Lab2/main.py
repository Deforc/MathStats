import scipy.stats as sps
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
position_characteristics_for_all_N = {}
for N in {20, 100, 500}:
    position_characteristics_for_all_N[N] = pd.DataFrame({"sample_average" : [],
                                   "sample_median" : [],
                                   "half_sum_extreme_elements" : [],
                                   "half_sum_quartiles" : [],
                                   "truncated_average" : []})
    for i in range(1000):
        temp_position_characteristics = []
        sample = sps.norm.rvs(size=N)
        sample.sort()
        temp_position_characteristics.append(np.mean(sample))
        temp_position_characteristics.append(np.median(sample))
        temp_position_characteristics.append((sample[0] + sample[-1]) / 2)
        temp_position_characteristics.append((np.quantile(sample, 0.25) + np.quantile(sample, 0.75)) / 2)
        temp_position_characteristics.append(np.mean(sample[int(N * 0.25) : int(N * 0.75)]))

        position_characteristics_for_all_N[N].loc[len(position_characteristics_for_all_N[N].index)] = temp_position_characteristics
results = pd.DataFrame()
for N in {20, 100, 500}:
    results = pd.concat([results, position_characteristics_for_all_N[N].
                        agg(["mean", lambda x: x.var(ddof=0)]).
                        rename(index={"mean":f"E(z) for N = {N}", "<lambda>":f"D(z) for N = {N}"})])
results.to_excel("results.xlsx")