#include "fastme.h"
#include <iostream>

extern "C" {

    int* test (double* d, int* init_adj, int n_taxa, int m) {
    double ** D = new double*[n_taxa];
    for(int i = 0; i< m; i++) D[i] = &d[i*n_taxa];

    int ** A = new int*[m];
    for(int i = 0; i< m; i++) A[i] = &init_adj[i*m];

    for(int i = 0; i< n_taxa; i++){
        for(int j=0; j<n_taxa; j++) std::cout<<D[i][j]<<" ";
        std::cout<<std::endl;
    }

    for(int i = 0; i< m; i++){
        for(int j=0; j<m; j++) std::cout<<A[i][j]<<" ";
        std::cout<<std::endl;
    }


    int** adj_mat = run(D, A, n_taxa, m);
    std::cout<<"gogogogo"<<std::endl;


    int* return_adj_mat = new int[m*m];
    for(int i = 0; i < m; i++){
        for(int j=0; j<m; j++) return_adj_mat[i*m +j] = adj_mat[i][j];
    }
    return return_adj_mat;
    }

}