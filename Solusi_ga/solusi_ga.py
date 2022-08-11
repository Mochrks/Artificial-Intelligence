import numpy
import ga

# Input persamaan.
equation_inputs = [4,-2,3.5,5,-11,-4.7]

# Banyaknya weight yang akan dioptimasi
num_weights = len(equation_inputs)

sol_per_pop = 8
num_parents_mating = 4

# Mendefinisikan ukuran populasi.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
# Membuat populasi awal.
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

best_outputs = []
num_generations = 1000
for generation in range(num_generations):
    print("Generation : ", generation)
    # Menghitung masing-masing fitness kromosom pada populasi.
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)
    print("Fitness")
    print(fitness)
    best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs,axis=1)))
    # Hasil terbaik dari current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs,axis=1)))
    # Memilih induk terbaik untuk proses melahirkan individu baru.
    parents = ga.select_mating_pool(new_population, fitness,num_parents_mating)

    print("Parents")
    print(parents)
    # Menghasilkan individu baru dari crossover.
    offspring_crossover = ga.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    print("Crossover")
    print(offspring_crossover)
    # Proses mutasi.
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)
    # Membuat populasi baru berdasarkan induk dan keturunan yang telah ada.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Mendapatkan solusi terbaik setelah selesai semua iterasi satu generasi
# Pertama, fitness dihitung untuk setiap solusi pada generasi terakhir
fitness = ga.cal_pop_fitness(equation_inputs, new_population)
# Kemudian kembalikan index yang menunjuk pada nilai fitness terbaik

best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()