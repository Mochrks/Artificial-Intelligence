import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Menghitung nilai fitness pada setiap solusi di current population
    # Fungsi fitness menghitung jumlah antara setiap input dan bobot yang menempeldengan inputnya

    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Memilih individu terbaik pada generasi yang sedang berjalan prosesnya sebagai induk untuk menghasilkan keturunan untuk generasi selanjutnya
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
        return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # memilih posisi crossover, biasanya ditengah-tengah kromosom.
    
    crossover_point = numpy.uint8(offspring_size[1]/2)
    
    for k in range(offspring_size[0]):
        # Index induk pertama.
        parent1_idx = k%parents.shape[0]
        # Index induk kedua.
        parent2_idx = (k+1)%parents.shape[0]
        # Keturunan yang baru akan memiliki setengah dari gen induk pertama
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Keturunan yang baru akan memiliki setengah dari gen induk kedua
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
    
def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Proses mutasi dengan bilangan random
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # Random value ditambahkan ke gen.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
            return offspring_crossover