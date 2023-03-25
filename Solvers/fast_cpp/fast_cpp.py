import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class Result(ctypes.Structure):
    _fields_ = [
        ('nni_counts', ctypes.c_int),
        ('spr_counts', ctypes.c_int),
        ('solution_adjs', ctypes.POINTER(ctypes.c_int)),
        ('objs', ctypes.POINTER(ctypes.c_double)),
    ]


class FastCpp:

    def __init__(self):
        self.lib = ctypes.CDLL('Solvers/fast_cpp/bridge.so.')
        self.numProcs = os.cpu_count()
        # self.lib.test.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        #                           ctypes.c_int, ctypes.c_int]
        self.lib.test.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                  ctypes.c_int, ctypes.c_int]

        self.lib.test_parallel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.test_obj.argtypes = []
        self.lib.test_obj.restype = ctypes.POINTER(Result)
        self.lib.test_parallel.restype = ctypes.POINTER(Result)

    def test(self, d, adj_mat, n_taxa):
        np.savetxt("Solvers/fast_cpp/mat", d, fmt='%.19f', delimiter=' ')
        np.savetxt("Solvers/fast_cpp/init_mat", adj_mat, fmt='%i', delimiter=' ')
        n = np.array([n_taxa], dtype=np.int32)
        np.savetxt("Solvers/fast_cpp/n_taxa", n, fmt='%i', delimiter=' ')
        os.system("Solvers/fast_cpp/fast_me")
        return np.loadtxt("Solvers/fast_cpp/result_adj_mat.txt", dtype=int)

    def run(self, d, adj_mat, n_taxa, m):
        self.lib.test.restype = ndpointer(dtype=ctypes.c_int32, shape=(m, m))
        adj = self.lib.test(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            adj_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            ctypes.c_int(n_taxa), ctypes.c_int(m))
        return adj

    def run_parallel(self, d, adj_mats, n_taxa, m, population_size):
        run_results = self.lib.test_parallel(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             adj_mats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                             ctypes.c_int(n_taxa), ctypes.c_int(m), ctypes.c_int(population_size),
                                             ctypes.c_int(self.numProcs))

        adjs = np.ctypeslib.as_array(run_results.contents.solution_adjs, shape=adj_mats.shape)
        objs = np.array(run_results.contents.objs[:population_size])
        nni_counts = run_results.contents.nni_counts
        spr_counts = run_results.contents.spr_counts

        return adjs, objs, nni_counts, spr_counts

    def test_return_object(self):
        test = self.lib.test_obj()
        objs = np.array(test.contents.objs[:10])
        adjs = np.array(test.contents.solution_adjs[:10])

        p = 0
