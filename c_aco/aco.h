#pragma once

#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

//__declspec(dllexport) size_t* step(double** closeness_matrix, double** pheromone_matrix,
//    const size_t node_count, const size_t ant_count,
//    const double A, const double B, const double Q, const double evap, double* bpl);
//
//__declspec(dllexport) void init_rand(time_t t);


size_t* step(double** closeness_matrix, double** pheromone_matrix,
    const size_t node_count, const size_t ant_count,
    const double A, const double B, const double Q, const double evap, double* bpl);
 
