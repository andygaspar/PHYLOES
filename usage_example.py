import numpy as np

from Solvers.PhyloES.phyloes import PhyloES


# distance matrix as numpy array
d = np.loadtxt('Data_/benchmarks/matrices/experiment_mats/100_rdpii_F81.csv')
# d = np.abs(np.around(np.loadtxt('Data_/benchmarks/matrices/experiment_mats/100_rdpii.csv'), 10))
# reducing d size to 30x30 for the example purposes
d = d[:30, :30]

# init phyloes
pes = PhyloES(d)

# run the algorithm
pes.solve()

# get solution tree length
print(pes.obj_val)

# get solution tree adj mat
print(pes.solution)

g = pes.solution