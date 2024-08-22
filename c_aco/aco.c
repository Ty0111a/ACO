#include "aco.h"
#include <omp.h>

int rand_int(int min, int max) {
    assert(min <= max);
    return rand() % (max - min + 1) + min;
}

bool contains(size_t* arr, size_t size, size_t val) {
    for (size_t i = 0; i < size; i++) {
        if (arr[i] == val) {
            return 1;
        }
    }
    return 0;
}

size_t* setdiff1d(size_t* a, size_t* b, size_t size_a, size_t size_b, size_t* size_r) {
    size_t count = 0;
    for (size_t i = 0; i < size_a; i++) {
        if (!contains(b, size_b, a[i])) {
            count++;
        }
    }
    size_t* result = malloc((size_t)(sizeof(size_t) * count));
    if (result == NULL) {
        free(a);
        return NULL;
    }
    size_t index = 0;
    for (size_t i = 0; i < size_a; i++) {
        if (!contains(b, size_b, a[i])) {
            result[index++] = a[i];
        }
    }
    *size_r = count;
    free(a);
    return result;
}

size_t* nonzero_index(double* array, size_t size, size_t* size_r) {
    size_t count = 0;
    for (size_t i = 0; i < size; i++) {
        if (array[i] != 0) {
            count++;
        }
    }
    size_t* result = malloc((size_t)(sizeof(size_t) * count));
    if (result == NULL) return NULL;
    size_t index = 0;
    for (size_t i = 0; i < size; i++) {
        if (array[i] != 0) {
            result[index++] = i;
        }
    }
    *size_r = count;
    return result;
}

double sum(double* arr, size_t size) {
    double sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

void power(double* arr, size_t size, double exponent) {
    for (size_t i = 0; i < size; i++) {
        arr[i] = pow(arr[i], exponent);
    }
}

void multiply_elements(double* a, double* b, size_t size) {
    assert(a != NULL && b != NULL);
    for (size_t i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}

size_t random_choice(double* probabilities, size_t size) {
    double random_value = (double)rand() / RAND_MAX * sum(probabilities, size);
    double cumulative_probability = 0;

    for (size_t i = 0; i < size; i++) {
        cumulative_probability += probabilities[i];
        if (random_value <= cumulative_probability) {
            return i;
        }
    }
    return size - 1;
}


size_t* ant_step(double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count,
    const double A, const double B, const double Q, const double E, double* bpl) {
    // simulate ant colony
    size_t** paths = (size_t**)malloc(ant_count * sizeof(size_t*));
    double* lens = malloc((double)(sizeof(double) * ant_count));

    if (paths == NULL || lens == NULL) {
        if (paths != NULL) {
	    free(paths);
	}
        if (lens != NULL) free(lens);
	printf("better path, paths, lens malloc error\n");
        return NULL;
    }

    for (size_t i = 0; i < ant_count; i++) {
        paths[i] = (size_t*)malloc(node_count * sizeof(size_t));
	lens[i] = LONG_MAX;
    }

    // num of threads 
    //omp_set_num_threads(g_nNumberOfThreads);
    int ant;
    #pragma omp parallel for private(ant) 
    for (ant = 0; ant < ant_count; ant++) {
        size_t* path = malloc((size_t)(sizeof(size_t) * node_count));
        if (path == NULL) {
	    for (size_t i = 0; i < ant_count; i++)
                free(paths[i]);
            free(paths);
            free(lens);
	    printf("path malloc error\n");
            continue;
        }
        path[0] = rand_int(0, node_count - 1);
        size_t p = 0;

        for (size_t i = 1; i < node_count; i++) {
            size_t size_enable;
            size_t* enable = nonzero_index(closeness_matrix[path[i - 1]], node_count, &size_enable);

            enable = setdiff1d(enable, path, size_enable, i, &size_enable);

            if (size_enable == 0) break;

            double* n = (double*)malloc((size_enable) * sizeof(double));
            double* t = (double*)malloc((size_enable) * sizeof(double));
            if (n == NULL || t == NULL || enable == NULL) {
		for (size_t i = 0; i < ant_count; i++)
	            free(paths[i]); 
                free(paths);
                free(lens);
                free(path);
                if (n != NULL) free(n);
                if (t != NULL) free(t);
		if (enable != NULL) free(enable);
		printf("n, t malloc error");
                continue; 
            }
            for (size_t j = 0; j < size_enable; j++) {
                n[j] = closeness_matrix[path[i - 1]][enable[j]];
                t[j] = pheromone_matrix[path[i - 1]][enable[j]];
            }
            power(n, size_enable, B);
            power(t, size_enable, A);

            multiply_elements(n, t, size_enable);
            free(t);

            size_t index = random_choice(n, size_enable);
            free(n);
            path[i] = enable[index];
            p++;
            free(enable);
        }

        // validator
        if ((p < node_count - 1) || (closeness_matrix[path[node_count - 1]][path[0]] == 0)) {
	    free(path);
            continue;
        }

        // calculate path len
        double path_len = 0;
        for (size_t i = 0; i < node_count - 1; i++) {
            path_len += 200 / closeness_matrix[path[i]][path[i + 1]];
        }   path_len += 200 / closeness_matrix[path[0]][path[node_count - 1]];

        for (size_t r = 0; r < node_count; r++) {
            paths[ant][r] = path[r];
        }
        lens[ant] = path_len;

        free(path);
    }
    
    // find min in paths
    double better_path_len = LONG_MAX;
    size_t index_min = 0;
    for (size_t i = 0; i < ant_count; i++) 
	if (better_path_len > lens[i]) {
            better_path_len = lens[i];
	    index_min = i;
	}

    // correct path not found
    size_t* better_path = malloc((size_t)(sizeof(size_t) * node_count));
    if (better_path == NULL) {
        for (size_t i=0; i < ant_count; i++)
	    free(paths[i]);
	free(paths);
	free(lens);
    }
    for (size_t i = 0; i < node_count; i++)
        better_path[i] = paths[index_min][i];
    *bpl = better_path_len;

    for (size_t i = 0; i < ant_count; i++)
	free(paths[i]);
    free(paths);
    free(lens);

    if (better_path_len == LONG_MAX) {
	free(better_path);
	return NULL;
    }   
    // evaporation
    double e = 1 - E;
    for (size_t i = 0; i < node_count; i++)
        for (size_t j = 0; j < node_count; j++)
            pheromone_matrix[i][j] *= e;

    // add ph
    double ph = Q / better_path_len;
    for (size_t i = 0; i < node_count - 1; i++) {
        pheromone_matrix[better_path[i]][better_path[i + 1]] += ph;
        pheromone_matrix[better_path[i + 1]][better_path[i]] += ph;
    }
    pheromone_matrix[better_path[0]][better_path[node_count - 1]] += ph;
    pheromone_matrix[better_path[node_count - 1]][better_path[0]] += ph;

    // max-min realization 
    for (size_t i = 0; i < node_count; i++)
        for (size_t j = 0; j < node_count; j++)
            if (pheromone_matrix[i][j] > 1) 
                pheromone_matrix[i][j] = 1;

    return better_path;
}

void free_better_path(size_t* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

size_t* run(double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count, const double A, const double B, const double Q, const double E, const size_t k, double* best_len) {
    size_t* best_path = (size_t*)malloc(node_count * sizeof(size_t));
    if (!best_path) {
        perror("Failed to allocate memory for best_path");
        return NULL;
    }
    *best_len = LONG_MAX;

    for (size_t i = 0; i < k; i++) {
        double current_len = LONG_MAX;
        size_t* current_path = ant_step(closeness_matrix, pheromone_matrix, node_count, ant_count, A, B, Q, E, &current_len);
        if (!current_path) {
            continue;
        } else if (current_len < *best_len) {
            *best_len = current_len;
            for (size_t x = 0; x < node_count; x++) {
                best_path[x] = current_path[x];
            }
        }
        free_better_path(current_path);
    }
    /*for (size_t t = 0; t < node_count; t++)
        printf("%ld\n", best_path[t]);*/
    return best_path;
}
