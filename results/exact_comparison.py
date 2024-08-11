import numpy as np
import pandas as pd

df = pd.read_csv('exact_comparison.csv')

df['fast_me_gap'] = df.fast/df.solver - 1
df['phyloes_gap'] = df.phyloes/df.solver - 1
df['run'] = range(1, 31)

df_fast_me = df[['run', 'n', 'fast', 'solver', 'f_time', 'solver_time', 'fast_me_gap']]
ddd = df_fast_me.style.format(precision=3, subset=['f_time', 'solver_time']).hide(axis="index")
print(ddd.to_latex())

df_opt = df_fast_me[df_fast_me.solver_time < 21000]

perc_time = 1 - df_opt.f_time/df_opt.solver_time
print(perc_time.mean(), perc_time.std())

df.columns
#phyloes

ddf = df[['run', 'n', 'fast', 'solver', 'phyloes', 'f_time', 'solver_time', 'phyloes_time', 'fast_me_gap', 'phyloes_gap']]
ddd = ddf.style.format(precision=3, subset=['phyloes_time', 'f_time', 'solver_time']).hide(axis="index")
print(ddd.to_latex())


df_opt_all = ddf[ddf.solver_time < 21000]

perc_time = 1 - df_opt_all.phyloes_time/df_opt_all.solver_time
print(perc_time.mean(), perc_time.std())

