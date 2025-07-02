import os
import random
import statistics
import time

from ACO import ACO, Graph

def main():
    dimensity = 2
    node_count = 100
    graph_number = ""

    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', f'{dimensity}d{node_count}{graph_number}.txt')

    parameters = {
        #"k_nearest": [36],
        "k_nearest": range(10, node_count+1, 10),
        #"ant_count": [43],
        "ant_count": range(10, node_count*3, 10),
        #"A": [0.51],
        "A": [i / 100 for i in range(1, 1000, 10)],
        "B": [i / 100 for i in range(1, 1000, 10)],
        "Q": range(20, node_count*10, 25),
        "evap": [i / 10 for i in range(1, 10)],
        #"evap": [0.2], 
        "start_ph": [i / 10 for i in range(1, 10)],
        #"start_ph": [0.2],
    }

    k = int(0.25 * node_count + 5)
    delta = 0 

    aco_launchs = 1 
    lenth_weight = 0.6
    time_weight = 1 - lenth_weight

    DE_generations = 10
    F = 0.6
    CR = 0.5
    population_size = 20 
    population = []

    best_performance = float('inf')
    best_info = ''

    ''' first DE generation '''
    find_solution_in_generation = 0
    # set params
    graph.load(file_path)
    for i in range(population_size):
        individual = {}
        for key in parameters:
            if key == "A":
                individual[key] = [random.choice(parameters[key])] * node_count
            else:
                individual[key] = random.choice(parameters[key])
 
        population.append(individual)

    start_time = time.time()
    # run first gen
    first_runs = []
    for individual in population:
        graph.load(file_path)
        graph.add_k_nearest_edges(int(individual['k_nearest']))

        aco = ACO(graph)
        individual_runs = [aco.run(ant_count=int(individual["ant_count"]), A=individual["A"], B=individual["B"], Q=individual["Q"], 
                                   E=individual["evap"], start_ph=individual["start_ph"], k=k, delta=delta, 
                                   max_generations=int(1.18 * node_count + 120)) for _ in range(aco_launchs)]  
        first_runs.append(min(individual_runs, key=lambda x: x[0][0]))
        individual['run_info'] = individual_runs
        individual['best_len'] = individual_runs[0][0][0]
        if individual['best_len'] < best_performance:
            best_performance = individual['best_len']
            find_soluton_in_generation = 0 
            best_info = individual_runs[0]

    # find uniform params
    first_runs.sort(key=lambda x: x[0][0])
    best_first_runs = first_runs[:int(len(first_runs) * 0.5)] # TODO по другому отсекать невалидные решения
    best_first_lenths = sorted([i[0][0] for i in best_first_runs])
    best_first_times = sorted([i[1] for i in best_first_runs])
    lenth_offset = min(best_first_lenths) 
    lenth_factor = (statistics.median(best_first_lenths) - lenth_offset) * 2
    time_offset = min(best_first_times) 
    time_factor = (statistics.median(best_first_times) - time_offset) * 2 

    # set rate for first generaion 
    for individual in population:
        all_performance = [lenth_weight * ((i[0][0] - lenth_offset) / lenth_factor) + 
                           time_weight * ((i[1] - time_offset) / time_factor) for i in individual["run_info"]]
        all_performance.sort()
        individual["performance"] = statistics.median(all_performance)
    population.sort(key=lambda x: x["performance"])

    #print(f"Generation 0: Best len = {best_performance:.2f}")
    
    ''' rum other DE generarions '''
    for generation in range(1, DE_generations+1):
        # print best
        best_ind = min(population, key=lambda x: x["performance"])
        #for best_ind in population:
            #print(f"{generation-1} {best_ind['k_nearest']} {best_ind['ant_count']} {best_ind['A'][0]} {best_ind['B']} {best_ind['Q']} {best_ind['evap']} {best_ind['start_ph']} {best_ind['performance']}")
        # mutate
        offspring = []
        for i in range(population_size):
            target = population[i]  # родительская особь
            a, b, c = random.sample([x for x in population if x != target], 3)
            mutant = {}
            for key in parameters:
                if key == "A":
                    mutant_val = [
                        (min_val + a_val)/2 if (val := a_val + F*(b_val - c_val)) < min_val else
                        (max_val + a_val)/2 if val > max_val else val
                        for a_val, b_val, c_val, min_val, max_val in 
                        zip(a[key], b[key], c[key], [min(parameters[key])]*node_count, [max(parameters[key])]*node_count)
                    ]
                else:
                    val = a[key] + F*(b[key] - c[key])
                    min_val, max_val = min(parameters[key]), max(parameters[key])
                    mutant_val = (min_val + a[key])/2 if val < min_val else \
                                  (max_val + a[key])/2 if val > max_val else val
            
                    if isinstance(parameters[key], range):
                        mutant_val = round(mutant_val)

                # Скрещивание с родителем (кроссовер)
                if random.random() < CR:
                    mutant[key] = mutant_val
                else:
                    mutant[key] = target[key]
 
            offspring.append(mutant)
        

        new_population = []
        for idx, individual in enumerate(offspring):
            # Загрузка и подготовка графа
            graph.load(file_path)
            graph.add_k_nearest_edges(int(individual['k_nearest']))
       
            # Запуск ACO
            aco = ACO(graph)
            runs = [aco.run(
                 ant_count=int(individual["ant_count"]),
                A=individual["A"],
                B=individual["B"],
                Q=individual["Q"],
                E=individual["evap"],
                start_ph=individual["start_ph"],
                k=k,
                delta=delta,
                max_generations=int(1.18 * node_count + 120)
            ) for _ in range(aco_launchs)]
        
            best_run = min(runs, key=lambda x: x[0][0])
            individual['run_info'] = runs
            individual['best_len'] = best_run[0][0]
            individual['time'] = best_run[1]
            
            # Расчет производительности
            len_norm = (individual['best_len'] - lenth_offset) / lenth_factor
            time_norm = (individual['time'] - time_offset) / time_factor
            individual['performance'] = lenth_weight * len_norm + time_weight * time_norm
        
            # Отбор: лучший между потомком и родителем
            if individual['performance'] < population[idx]['performance']:
                new_population.append(individual)
                if individual['best_len'] < best_performance:
                    find_solution_in_generation = generation
                    best_performance = individual['best_len']
                    best_solution = individual
            else:
                new_population.append(population[idx])
    
        population = new_population
        #print(f"Generation {generation}: Best len = {best_performance:.2f}")

    #print(best_info) 
    print(f"{find_solution_in_generation} {best_performance}")


if __name__ == "__main__":
    for _ in range(49):
        main()
