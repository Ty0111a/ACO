#include <stdlib.h>
#include <float.h>
#include <omp.h>

void swap(size_t* a, size_t* b) {
    size_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline void check_path(size_t* arr, double** distance_matrix, size_t node_count, double* min_len, size_t* best_path) {
    // Пропускаем обратные пути (оптимизация для симметричного TSP)
    if (arr[1] > arr[node_count-1]) return;

    double len = 0;
    // Рассчитываем длину пути (все рёбра гарантированно существуют)
    for (size_t i = 0; i < node_count - 1; ++i) {
        len += distance_matrix[arr[i]][arr[i + 1]];
    }
    len += distance_matrix[arr[node_count-1]][arr[0]];  // Замыкающее ребро

    #pragma omp critical
    {
        if (len < *min_len) {
            *min_len = len;
            for (size_t i = 0; i < node_count; ++i)
                best_path[i] = arr[i];
        }
    }
}

void permute(size_t* arr, size_t l, size_t r, double** distance_matrix, size_t node_count, double* min_len, size_t* best_path) {
    if (l == r) {
        check_path(arr, distance_matrix, node_count, min_len, best_path);
        return;
    }

    for (size_t i = l; i <= r; i++) {
        swap(&arr[l], &arr[i]);
        permute(arr, l + 1, r, distance_matrix, node_count, min_len, best_path);
        swap(&arr[l], &arr[i]);
    }
}

size_t* run_brute_force(double** distance_matrix, size_t node_count, double* best_len) {
    size_t* best_path = malloc(sizeof(size_t) * node_count);
    if (!best_path) return NULL;

    *best_len = DBL_MAX;

    if (node_count == 0) return best_path;
    if (node_count == 1) {
        *best_len = 0;
        best_path[0] = 0;
        return best_path;
    }

    #pragma omp parallel
    {
        size_t* path = malloc(sizeof(size_t) * node_count);
        size_t* local_best = malloc(sizeof(size_t) * node_count);
        double local_min_len = DBL_MAX;

        #pragma omp for
        for (size_t i = 0; i < node_count; i++) {
            // Инициализация базового маршрута
            for (size_t k = 0; k < node_count; k++) path[k] = k;
            
            // Фиксируем первую вершину
            swap(&path[0], &path[i]);
            
            // Перебираем только вершины, меньшие чем последняя
            for (size_t j = 1; j < node_count; j++) {
                if (j == i) continue;
                
                swap(&path[1], &path[j]);
                permute(path, 2, node_count - 1, distance_matrix, node_count, &local_min_len, local_best);
                swap(&path[1], &path[j]);  // Возвращаем на место
            }
        }

        #pragma omp critical
        {
            if (local_min_len < *best_len) {
                *best_len = local_min_len;
                for (size_t k = 0; k < node_count; k++)
                    best_path[k] = local_best[k];
            }
        }

        free(path);
        free(local_best);
    }

    return best_path;
}

void free_brute_result(size_t* ptr) {
    if (ptr) free(ptr);
}
