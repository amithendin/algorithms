'''
Simple genetic algorithm
by: Amit Hendin
course: 20551 Intro to Artificial Intelligence Open University of Israel

the following program finds an array of 10 numbers that fit a simple criteria specified in the fit() function
using a basic genetic algorithm
'''
import random

def mul(arr):
    mx = 1
    for i in arr:
        mx *= i
    return mx

def fit(x):
    sx = sum(x[0:5])
    mx = mul(x[5:10])
    return 1 / (abs(36 - sx) + abs(360 - mx) + 1)

def selection(pop):
    chosen = list(pop.copy())
    chosen.sort(key=fit)
    return chosen[0:5]

def crossover(chosen):
    pop = []

    for i in range(5):
        a,b = random.sample(chosen,2)
        c = []
        if abs(36 - sum(a[0:5])) < abs(36 - sum(b[0:5])):
            c += a[0:5]
        else:
            c += b[0:5]

        if abs(360 - mul(a[5:10])) < abs(360 - mul(b[5:10])):
            c += a[0:5]
        else:
            c += b[0:5]

        pop.append(c)

    nums = list(range(1,11))
    for i in range(5):
        pop.append(random.sample(nums, 10))

    return pop

def run():
    nums = list(range(1,11))
    pop = []
    for i in range(10):
        pop.append(random.sample(nums, 10))

    gens = 0
    while gens < 1000:
        for x in pop:
            if fit(x) == 1:
                print('found',x, 'after',gens,'generations')
                return x
        chosen = selection(pop)
        pop = crossover(chosen)
        gens += 1

run()