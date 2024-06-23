import numpy as np
import tqdm


# gene - is a (x,y) tuple with is a location on the grid in which to put a 1
# search space - is the space of all arrays of tuples (x,y) where all the tuples in the array are different and the array length between 10 and n
# crossover - is combining sections of arrays together where a section is defined by the euclidean distance between it's cells meaning cell which are close together
# comprise a section
# mutation - is adding a random amount of random coordiates to the array
#######################################
# generation - in GA is generation in CA
# the ones that grow well reproduce
# the ones that don't die

num_generations = 500
num_replaced = 0
#fitness_over_generations = np.zeros(num_generations)
evaluation_score_over_generations = np.zeros(num_generations)

def apply_genes_to_grid(individual, individual_grid):
    individual_grid[:, :] = 0
    for gene in individual:
        individual_grid[gene[0], gene[1]] = 1


def population_genesis(grid_size, population_size, gene_per_instance):
    population = np.random.randint(grid_size, size=(population_size, gene_per_instance, 2), dtype=np.int8)
    population_grid = np.zeros((population_size, grid_size, grid_size), dtype=np.int8)

    for i in range(population_size):
        apply_genes_to_grid(population[i], population_grid[i])

    return population, population_grid


def automata_step(individual_grid):
    new_grid = np.zeros(individual_grid.shape)

    for x in range(individual_grid.shape[0]):
        for y in range(individual_grid.shape[1]):
            m = individual_grid[x, y]
            sum_neighbors = np.sum(individual_grid[x - 1:x + 2, y - 1:y + 2]) - m

            if m == 1 and sum_neighbors == 2 or sum_neighbors == 3:
                new_grid[x, y] = 1
            if m == 0 and sum_neighbors == 3:
                new_grid[x, y] = 1

    return new_grid


def hash_grid(population_grid):
    pack = np.packbits(population_grid, axis=1)
    return pack.reshape(pack.shape[0], pack.shape[1] * pack.shape[2])


def evaluate(population_grid, new_population_grid, population_age, min_age, grid_hash, new_grid_hash):
    flat_grid = population_grid.reshape(population_grid.shape[0], population_grid.shape[1] ** 2)
    new_flat_grid = new_population_grid.reshape(population_grid.shape[0], population_grid.shape[1] ** 2)

    sum_grid = np.sum(flat_grid, axis=1)
    sum_new_grid = np.sum(new_flat_grid, axis=1)
    evaluation_score = sum_new_grid - sum_grid

    is_cycle = np.all(grid_hash == new_grid_hash, axis=1)
    is_static = np.all(flat_grid == new_flat_grid, axis=1)
    young_stable = np.logical_and( np.logical_or(is_static, is_cycle), population_age < min_age)

    evaluation_score[evaluation_score == 0] = int(0.1*(population_grid.shape[1] ** 2))+1
    evaluation_score[sum_new_grid == 0] = 0
    evaluation_score[young_stable] = 0  # stabilized too early

    return evaluation_score, is_static


def normalize(x):
    norm_x = x #(x - np.mean(x)) / np.std(x)
    if np.min(norm_x) < 0:
        norm_x = norm_x - np.min(norm_x)
    sum_norm_x = np.sum(norm_x)
    prob_x = norm_x
    if sum_norm_x != 0:
        prob_x /= sum_norm_x
    return prob_x


def fitness(evaluation_score, grid_size):
    fitness_score = np.zeros(evaluation_score.shape)

    shrunk_index = evaluation_score < 0
    grown_or_stable_index = evaluation_score > 0

    fitness_score[shrunk_index] = 1 / (-evaluation_score[shrunk_index])
    fitness_score[grown_or_stable_index] = evaluation_score[grown_or_stable_index] / (grid_size ** 2)

    return normalize(fitness_score)


def crossover(individual1, individual2):
    # quarter = int(individual1.shape[0] / 4)
    # first_section = individual1[0: quarter]
    # second_section = individual2[quarter: quarter * 2]
    # third_section = individual1[quarter * 2: quarter * 3]
    # forth_section = individual2[quarter * 3:]
    #
    # return np.concatenate([first_section, second_section, third_section, forth_section])
    individual_size = individual1.shape[0]
    half = int(individual_size/2)
    idx = np.random.choice(individual_size, size=individual_size, replace=False)
    return np.concatenate([ individual1[ idx[:half] ], individual2[ idx[half:] ] ])


def mutate(individual, grid_size):
    num_genes_to_mutate = np.random.randint(low=1, high=int(individual.shape[0]/4))
    indexes_to_mutate = np.random.choice(individual.shape[0], size=num_genes_to_mutate, replace=False)
    individual[indexes_to_mutate] = np.random.randint(grid_size, size=(num_genes_to_mutate, 2))


def selection(fitness_scores, population_size):
    return np.random.choice(population_size, size=2, p=fitness_scores, replace=False)


def sort_by_fitness(fitness_scores):
    tmp = np.array([[i for i in range(fitness_scores.shape[0])], fitness_scores]).T
    sort_idx = np.argsort(tmp[:, 1])
    return sort_idx

def run_genetic_methuselah(RANDOM_SEED, GRID_SIZE, POPULATION_SIZE, NUM_GENERATIONS, MIN_METHUSELAH_AGE, MAX_CYCLE_AGE):
    global evaluation_score_over_generations
    global num_replaced
    GENE_PER_INSTANCE = int(0.1 * GRID_SIZE ** 2)
    MAX_AGE = int(MIN_METHUSELAH_AGE*1.1)
    np.random.seed(RANDOM_SEED)

    print("BEGIN METHUSELAH SEARCH")
    population, population_grid = population_genesis(GRID_SIZE, POPULATION_SIZE, GENE_PER_INSTANCE)
    population_age = np.zeros(POPULATION_SIZE)
    grid_hash = hash_grid(population_grid)

    for generation in tqdm.tqdm(range(NUM_GENERATIONS)):
        np.random.seed(RANDOM_SEED+generation)

        new_population_grid = []
        for i in range(POPULATION_SIZE):
            new_population_grid.append(automata_step(population_grid[i]))
        new_population_grid = np.array(new_population_grid, dtype=np.int8)

        new_grid_hash = hash_grid(new_population_grid)

        evaluation_scores, is_static = evaluate(population_grid, new_population_grid, population_age, MIN_METHUSELAH_AGE,
                                     grid_hash, new_grid_hash)
        fitness_scores = fitness(evaluation_scores, GRID_SIZE)

        cycle_need_refresh = population_age % MAX_CYCLE_AGE == 0
        grid_hash[cycle_need_refresh] = new_grid_hash[cycle_need_refresh]

        old_and_static = np.logical_and(population_age > MAX_AGE, is_static )
        dead_individuals = np.where( np.logical_or(fitness_scores == 0, old_and_static) )

        evaluation_score_over_generations[generation] = np.mean(evaluation_scores)

        if generation % 50 == 0:
            print(evaluation_score_over_generations[generation], num_replaced)


        if len(evaluation_scores[evaluation_scores != 0]) < 2:
            population, population_grid = population_genesis(GRID_SIZE, POPULATION_SIZE, GENE_PER_INSTANCE)
            population_age = np.zeros(POPULATION_SIZE)
            grid_hash = hash_grid(population_grid)
            num_replaced += POPULATION_SIZE
        else:

            population_grid = new_population_grid
            population_age = population_age + 1
            # replace the dead candidates
            for dead_index in dead_individuals[0]:
                cadidate1, candidate2 = selection(fitness_scores, POPULATION_SIZE)
                new_individual = crossover(population[cadidate1], population[candidate2])
                mutate(new_individual, GRID_SIZE)
                population[dead_index] = new_individual
                apply_genes_to_grid(population[dead_index], population_grid[dead_index])
                population_age[dead_index] = 0
                num_replaced += 1

    print("END METHUSELAH SEARCH")

    return population, population_age, fitness_scores


p, pa, f = run_genetic_methuselah(
    RANDOM_SEED=0,
    GRID_SIZE=32,
    POPULATION_SIZE=50,
    NUM_GENERATIONS=num_generations,
    MIN_METHUSELAH_AGE=50,
    MAX_CYCLE_AGE=4)

sort_idx = sort_by_fitness(f)
p = p[sort_idx]
f = f[sort_idx]
pa = pa[sort_idx]

p_shape_str = "".join([str(x) + "_" for x in p.shape])
p.tofile("methuselah_" + p_shape_str + 'grid_32_fit_'+"{:.4f}".format(np.max(f)).replace('.','_')+'.numpy')

import matplotlib.pyplot as plt

x = np.array([i for i in range(num_generations)])

plt.title("Evaluation scores over all generations")
plt.plot(x, evaluation_score_over_generations)
plt.show()

from ca_engine import CAEngine

test_grid = np.zeros((32,32))
apply_genes_to_grid(p[-1], test_grid)
print("methuselah age", pa[-1], "generations","with fitness ", f[-1])

CAEngine(
    cell_size=20, fps=10,
    grid=test_grid,
    animate=True
).start()
