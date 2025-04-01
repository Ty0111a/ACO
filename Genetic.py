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
        #"A": [0.41],
        "A": [i / 100 for i in range(1, 200, 10)],
        "B": range(3, 14),
        "Q": range(20, node_count*4, 25),
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

    genetic_generations = 10
    population_size = 30 
    elite_part = 0.5
    population = []

    best_run_performance = float('inf')
    best_run_info = ''

    ''' first genetic generation '''
    find_solution_in_generation = 0
    l = 0
    # set random params
    for i in range(population_size):
        individual = {}
        for key in parameters:
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
        if individual['best_len'] < best_run_performance:
            best_run_performance = individual['best_len']
            find_soluton_in_generation = l
            best_run_info = individual_runs[0]
            
        
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
    
    ''' rum other genetic generarions '''
    while True:
        #print(f"gen {l} in {time.time() - start_time:.0f} sec")
        #print(best_run_info)
        start_time = time.time() 
        # elitism
        elite_size = int(population_size * elite_part)
        elite = sorted(population, key=lambda x: x["performance"])[:elite_size]
        for individual in elite:
            # print(f"{l} {individual['k_nearest']} {individual['ant_count']} {individual['A']} {individual['B']} {individual['Q']} {individual['evap']} {individual['start_ph']} {individual['performance']}")
            pass

        l += 1
        if l > genetic_generations - 1: 
            #print(f"{individual['k_nearest']} {individual['ant_count']} {individual['A']} {individual['B']} {individual['Q']} {individual['evap']} {individual['start_ph']} {individual['performance']}")
            break # stop genetic

        def crossover(mommy, daddy):
            child = {}
            for j in ["k_nearest", "ant_count", "A", "B", "Q", "evap", "start_ph"]:
                minj = min(mommy[j], daddy[j])
                maxj = max(mommy[j], daddy[j])
                dmin = minj - 0.25 * (maxj - minj)
                dmax = minj + 0.25 * (maxj - minj)
                child[j] = random.uniform(dmin, dmax)

            for key in child:
                child[key] = round(abs(child[key]), 2)
            if child["evap"] > 0.99:
                child["evap"] = 0.99

            return child


        def mutate(child):
            for i in ["k_nearest", "ant_count", "A", "B", "Q", "evap", "start_ph"]:
                if random.random() < 0.1:
                    low = min(parameters[i])
                    high = max(parameters[i])
                    if low == high:
                        continue
                    mode = child[i] 
                    lambd = 60
                    alpha = max(0.00001, abs(1 + lambd * (mode - low) / (high - low)))
                    beta = max(0.00001 , abs(1 + lambd * (high - mode) / (high - low)))
                    child[i] = low + (high - low) * random.betavariate(alpha, beta)

            for key in child:
                child[key] = round(abs(child[key]), 2) 
            if child["evap"] > 0.99: child["evap"] = 0.99
            return child


        # crossing and mutate
        offspring = []

        for i in range(population_size - elite_size):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = crossover(parent1, parent2)
            child = mutate(child)
            offspring.append(child)

        # find performance of aco
        for individual in offspring:
            graph.load(file_path)
            graph.add_k_nearest_edges(int(individual['k_nearest']))
            aco = ACO(graph)
            all_runs = [
                aco.run(ant_count=int(individual["ant_count"]), A=individual["A"], B=individual["B"], Q=individual["Q"],
                                      E=individual["evap"], start_ph=individual["start_ph"], k=k, delta=delta,
                                      max_generations=int(1.18 * node_count + 120)) for _ in range(aco_launchs)
                       ]
            all_performance = [lenth_weight*((i[0][0]-lenth_offset)/lenth_factor)+time_weight*((i[1]-time_offset)/time_factor) for i in all_runs]
            all_performance.sort()
            individual["performance"] = statistics.median(all_performance)
            individual["best_len"] = all_runs[0][0][0]
            if individual['best_len'] < best_run_performance:
                best_run_performance = individual['best_len']
                best_run_info = all_runs[0]
                find_solution_in_generation = l

        population = elite + offspring
        population.sort(key=lambda x: x["performance"])  
        # print(min(population, key=lambda x: x["performance"]))

    #print(best_run_info) 
    print(f"{find_solution_in_generation} {best_run_performance}")


if __name__ == "__main__":
    for _ in range(500):
        main()
