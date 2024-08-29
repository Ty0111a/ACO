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


size_t* run_fixed_generation(double** closeness_matrix, double** pheromone_matrix, 
const size_t node_count, const size_t ant_count,
    const double A, const double B, const double Q, const double E, const size_t k, double* best_len);
 
size_t* run_until_repeated_solution(double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count, const double A, const double B, const double Q, const double E, const size_t k, double* best_len); 

size_t* run_until_stable_solution(double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count, const double A, const double B, const double Q, const double E, const size_t k, const double delta, double* best_len);

