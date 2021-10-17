import mlrose
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

## task1
### n-queen problem
def n_queens_compare(n, runs = 100):
    
    # define problem
    fitness = mlrose.Queens()
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = False, max_val = n)
    
    # randomly define initial states
    np.random.seed(2)
    RHC_accu = []
    SA_accu = []
    GA_accu = []
    MIMIC_accu = []
    
    for i in range(runs):
        init_state = np.random.randint(low = 0, high = n, size = n)
        RHC_state, RHC_fitness = mlrose.random_hill_climb(problem, max_attempts = 10, 
                                                      max_iters = 1000, restarts = 10,
                                                      init_state = init_state, curve = False, random_state = 1)
        
        schedule = mlrose.ExpDecay()
        SA_state, SA_fitness = mlrose.simulated_annealing(problem, schedule = schedule,
                                                              max_attempts = 10, max_iters = 1000,
                                                              init_state = init_state, random_state = 1)
        
        GA_state, GA_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)

        MIMIC_state, MIMIC_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)
        
        RHC_accu.append(RHC_fitness)
        SA_accu.append(SA_fitness)
        GA_accu.append(GA_fitness)
        MIMIC_accu.append(MIMIC_fitness)
    
    print(RHC_state, RHC_fitness, SA_state, SA_fitness, GA_state, GA_fitness, MIMIC_state, MIMIC_fitness)
    return [RHC_accu, SA_accu, GA_accu, MIMIC_accu]
 
res = n_queens_compare(8, runs = 1000)
print('RHC', np.mean(res[0]))
print('SA', np.mean(res[1]))
print('GA', np.mean(res[2]))
print('MIMIC', np.mean(res[3]))    

res1 = n_queens_compare(16, runs = 100)
print('RHC', np.mean(res1[0]))
print('SA', np.mean(res1[1]))
print('GA', np.mean(res1[2]))
print('MIMIC', np.mean(res1[3]))   


### 4-peak problem
def peaks_compare(n = 12, runs = 100):
    
    # define problem
    fitness = mlrose.FourPeaks(t_pct=0.15)
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True, max_val = 2)
    
    # randomly define initial states
    np.random.seed(2)
    RHC_accu = []
    SA_accu = []
    GA_accu = []
    MIMIC_accu = []
    
    for i in range(runs):
        init_state = np.random.randint(low = 0, high = 2, size = n)
        RHC_state, RHC_fitness = mlrose.random_hill_climb(problem, max_attempts = 10, 
                                                      max_iters = 1000, restarts = 10,
                                                      init_state = init_state, curve = False, random_state = 1)
        schedule = mlrose.ExpDecay()
        SA_state, SA_fitness = mlrose.simulated_annealing(problem, schedule = schedule,
                                                              max_attempts = 10, max_iters = 1000,
                                                              init_state = init_state, random_state = 1)
        
        GA_state, GA_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)

        MIMIC_state, MIMIC_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)
        
        RHC_accu.append(RHC_fitness)
        SA_accu.append(SA_fitness)
        GA_accu.append(GA_fitness)
        MIMIC_accu.append(MIMIC_fitness)
    
    print(RHC_state, RHC_fitness, SA_state, SA_fitness, GA_state, GA_fitness, MIMIC_state, MIMIC_fitness)
    return [RHC_accu, SA_accu, GA_accu, MIMIC_accu]
 
fitness = mlrose.FourPeaks(t_pct=0.15)
state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
fitness.evaluate(state)
# max 21
peaks_res = peaks_compare(12, runs = 1000)
print('RHC', np.mean(peaks_res[0]))
print('SA', np.mean(peaks_res[1]))
print('GA', np.mean(peaks_res[2]))
print('MIMIC', np.mean(peaks_res[3]))      

### Flipflop problem
def flipflop_compare(n = 12, runs = 100):
    
    # define problem
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True, max_val = 2)
    
    # randomly define initial states
    np.random.seed(2)
    RHC_accu = []
    SA_accu = []
    GA_accu = []
    MIMIC_accu = []
    
    for i in range(runs):
        init_state = np.random.randint(low = 0, high = 2, size = n)
        RHC_state, RHC_fitness = mlrose.random_hill_climb(problem, max_attempts = 10, 
                                                      max_iters = 1000, restarts = 10,
                                                      init_state = init_state, curve = False, random_state = 1)
        schedule = mlrose.ExpDecay()
        SA_state, SA_fitness = mlrose.simulated_annealing(problem, schedule = schedule,
                                                              max_attempts = 10, max_iters = 1000,
                                                              init_state = init_state, random_state = 1)
        
        GA_state, GA_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)

        MIMIC_state, MIMIC_fitness = mlrose.genetic_alg(problem, pop_size = 200, 
                                                mutation_prob = 0.1, max_attempts = 10, max_iters = 1000, 
                                                curve = False, random_state = 1)
        
        RHC_accu.append(RHC_fitness)
        SA_accu.append(SA_fitness)
        GA_accu.append(GA_fitness)
        MIMIC_accu.append(MIMIC_fitness)
    
    print(RHC_state, RHC_fitness, SA_state, SA_fitness, GA_state, GA_fitness, MIMIC_state, MIMIC_fitness)
    return [RHC_accu, SA_accu, GA_accu, MIMIC_accu]
        
flipflop_res = flipflop_compare(30, runs = 1000)
print('RHC', np.mean(flipflop_res[0]))
print('SA', np.mean(flipflop_res[1]))
print('GA', np.mean(flipflop_res[2]))
print('MIMIC', np.mean(flipflop_res[3]))


## task2

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
                                                    test_size = 0.2, random_state = 3)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

one_hot = OneHotEncoder()

y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

### RHC
RHC_NN = mlrose.NeuralNetwork(hidden_nodes = [100], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 10000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, restarts = 10, \
                                 max_attempts = 100, random_state = 3)

RHC_NN.fit(X_train_scaled, y_train)

y_train_accuracy = accuracy_score(y_train, RHC_NN.predict(X_train_scaled))

print(y_train_accuracy)

y_test_accuracy = accuracy_score(y_test, RHC_NN.predict(X_test_scaled))

print(y_test_accuracy)

### SA
SA_NN = mlrose.NeuralNetwork(hidden_nodes = [100], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 10000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, \
                                 max_attempts = 100, random_state = 3)

SA_NN.fit(X_train_scaled, y_train)

y_train_accuracy = accuracy_score(y_train, SA_NN.predict(X_train_scaled))

print(y_train_accuracy)

y_test_accuracy = accuracy_score(y_test, SA_NN.predict(X_test_scaled))

print(y_test_accuracy)

### GA
GA_NN = mlrose.NeuralNetwork(hidden_nodes = [100], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 10000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, \
                                 max_attempts = 100, random_state = 3)

GA_NN.fit(X_train_scaled, y_train)

y_train_accuracy = accuracy_score(y_train, GA_NN.predict(X_train_scaled))

print(y_train_accuracy)

y_test_accuracy = accuracy_score(y_test, GA_NN.predict(X_test_scaled))

print(y_test_accuracy)

### GD
GD_NN = mlrose.NeuralNetwork(hidden_nodes = [100], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 10000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, \
                                 max_attempts = 100, random_state = 3)

GD_NN.fit(X_train_scaled, y_train)

y_train_accuracy = accuracy_score(y_train, GD_NN.predict(X_train_scaled))

print(y_train_accuracy)

y_test_accuracy = accuracy_score(y_test, GD_NN.predict(X_test_scaled))

print(y_test_accuracy)