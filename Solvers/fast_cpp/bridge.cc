#include "fastme.h"
#include <iostream>
#include <omp.h>
#include <sched.h>

extern "C" {

    int* test (double* d, int* init_adj, int n_taxa, int m) {

        double ** D = new double*[n_taxa];
        int i,j;

        for(i = 0; i< n_taxa; i++) D[i] = &d[i*n_taxa];
 
        int ** A = new int*[m];
        for(i = 0; i< m; i++) A[i] = &init_adj[i*m];
                
        int* solution_mat = new int[m*m];
        run(D, A, solution_mat, n_taxa, m);

        if (D!= nullptr) delete[] D;
        if (A!= nullptr) delete[] A;

        return solution_mat;

        }




    int* test_parallel (double* d, int* init_adj, int n_taxa, int m, int population_size, int num_procs) {


        double ** D = new double*[n_taxa];
        int i,j,t, mat_size;
        mat_size = m*m;

        for(i = 0; i< n_taxa; i++) D[i] = &d[i*n_taxa];


        int *** A = new int**[population_size];
        int* solution_mat = new int[population_size*mat_size];


        omp_set_num_threads(num_procs);
        #pragma omp parallel for schedule(static) shared(d)
        for(t = 0; t < population_size; t++) {
            A[t]= new int*[m];
            for(i = 0; i< m; i++) A[t][i] = &init_adj[t*mat_size + i*m];
                
            
            run(D, A[t], &solution_mat[t*mat_size], n_taxa, m);
        }

        if (D!= nullptr) delete[] D;
        if (A!= nullptr) delete[] A;

        return solution_mat;
        }

}