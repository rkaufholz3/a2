# mlrose library is used, ref: Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package
# for Python. https://github.com/gkhayes/mlrose. Accessed: 24 Sep 2019.

import mlrose
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# warnings.filterwarnings("ignore")


def load_adult_data(train_size=0):

    # Reused from Assignment 1 (rkaufholz3), copied rather than import for simplicity, converted to Python 3.7...

    # Data source (Adult): https://archive.ics.uci.edu/ml/datasets/Adult

    # Note: Adult data is provided already split between training and test sets, in separate files.  These are merged
    # here, pre-processed as a single file, then split back into new Train / Test sets.

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'label']

    # Load and merge data
    train_data = pd.read_csv('./data/adult.data', header=None, names=column_names, skipinitialspace=True)
    test_data = pd.read_csv('./data/adult.test', header=None, names=column_names, skipinitialspace=True, skiprows=[0])
    all_data = train_data.append(test_data, ignore_index=True).reset_index(drop=True)

    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)
    print("all_data shape:", all_data.shape)
    print()

    # Segregate numerical from categorical features (dropping captital-gain and capital-loss as it's 0 for most
    numerical = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country']
    labels = ['label']
    selection = numerical + categorical + labels
    select_all_data = all_data[selection]

    # Drop instances with missing / poor data
    # print "Empty / null values:\n", select_all_data.isnull().any()  # no null values
    # print select_all_data[select_all_data.sex.isin(['Male'])]
    # print select_all_data[select_all_data.age.isnull()]
    # Remove inconsistent '.' from label
    # https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
    select_all_data['label'] = select_all_data['label'].map(lambda x: x.rstrip('.'))
    # Map '<=50K' to 0, '>50K' to 1
    select_all_data['label'] = select_all_data['label'].replace('<=50K', 0)
    select_all_data['label'] = select_all_data['label'].replace('>50K', 1)
    clean_data = select_all_data.dropna(axis=0).reset_index(drop=True)

    # clean_data.groupby('label').hist()
    # plt.show()

    print('\nclean_data shape:', clean_data.shape)
    print("\nduplicate rows count:", len(all_data[all_data.duplicated(selection)]))
    # print "\nduplicate rows:", all_data[all_data.duplicated(selection)]

    # Extract features and labels
    X = clean_data.drop('label', axis=1)
    y = clean_data['label']
    print('\nX shape:', X.shape)
    print('y shape:', y.shape)

    # Perform one-hot-encoding for categorical features
    encoded_X = pd.get_dummies(X, columns=categorical)
    print("encoded_X shape:", encoded_X.shape)
    print()

    # Split into Train and Test sets

    # Check class distribution before splitting train data
    # hist = y.hist()
    # plt.show()

    # Split data into training and testing sets (shuffled and stratified)
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.33, random_state=None, stratify=y,
                                                        shuffle=True)

    # X_train_sampled, y_train_sampled = X_train, y_train

    # Check train set sizes after sampling
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # Check class distribution after splitting train data
    # hist = y_train.hist()
    # plt.show()

    if train_size != 0:
        X_train_sampled, X_train_unused, y_train_sampled, y_train_unused = train_test_split(X_train, y_train,
                                                                                            train_size=train_size,
                                                                                            random_state=42,
                                                                                            stratify=y_train,
                                                                                            shuffle=True)
    else:
        X_train_sampled, y_train_sampled = X_train, y_train

    # Check train set sizes after sampling
    print('X_train_sampled shape:', X_train_sampled.shape)
    print('y_train_sampled shape:', y_train_sampled.shape)

    # Scale numerical features
    # https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-on-a-partial-part-of-a-data-set
    X_train_scaled = X_train_sampled.copy()
    X_test_scaled = X_test.copy()
    X_train_numerical = X_train_scaled[numerical]
    X_test_numerical = X_test_scaled[numerical]
    scaler = preprocessing.StandardScaler().fit(X_train_numerical)  # Fit using only Train data
    numerical_X_train = scaler.transform(X_train_numerical)
    numerical_X_test = scaler.transform(X_test_numerical)  # transform X_test with same scaler as X_train
    X_train_scaled[numerical] = numerical_X_train
    X_test_scaled[numerical] = numerical_X_test

    print("\nX_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Select important features based on correlation analysis
    # plot_correlation(X_train_scaled, y_train)
    # Features to keep: Corr. > 0.03 and Corr. < -0.05 (to start with), based on correlation plot.
    # And dropping 'sex_Female' given inverse correlation with 'sex_Male'
    # Current...
    features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
                        'age', 'sex_Male', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
                        'education_Bachelors', 'education_Masters', 'education_Prof-school', 'workclass_Self-emp-inc',
                        'education_Doctorate', 'relationship_Wife', 'race_White', 'workclass_Federal-gov',
                        'workclass_Local-gov', 'native-country_United-States', 'education_9th',
                        'occupation_Farming-fishing', 'education_Some-college', 'education_7th-8th',
                        'native-country_Mexico', 'marital-status_Widowed', 'education_10th',
                        'occupation_Machine-op-inspct', 'marital-status_Separated', 'workclass_Private',
                        'workclass_?', 'occupation_?', 'occupation_Adm-clerical', 'occupation_Handlers-cleaners',
                        'education_11th', 'relationship_Other-relative', 'race_Black', 'marital-status_Divorced',
                        'education_HS-grad', 'relationship_Unmarried', 'occupation_Other-service',
                        'relationship_Not-in-family', 'relationship_Own-child',
                        'marital-status_Never-married']

    final_X_train = X_train_scaled[features_to_keep]
    final_X_test = X_test_scaled[features_to_keep]

    # plot_correlation(final_X_train, y_train_sampled)

    print("\nfinal_X_train shape:", final_X_train.shape)
    print("final_X_test shape:", final_X_test.shape)
    print("y_train shape:", y_train_sampled.shape)
    print("y_test shape:", y_test.shape)
    print()

    return final_X_train, final_X_test, y_train_sampled, y_test


def evaluate_solution_space(n, fitness_fn, plot, verbose):

    # Generate a list of all combinations for n binary values
    # https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
    lst = list(itertools.product([0, 1], repeat=n))

    # Generate a list of fitness scores for all combinations
    fitness_scores = []
    for i in lst:
        state = np.array(i)
        fitness_scores.append(fitness_fn.evaluate(state))
        # print(i, + fitness_fn.evaluate(state))

    # Determine results
    # https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
    unique, counts = np.unique(fitness_scores, return_counts=True)
    results = dict(zip(unique, counts))
    true_max = np.amax(fitness_scores)
    num_max = results[np.amax(fitness_scores)]

    if verbose:
        print()
        print('Bits:', n)
        print('Counts by fitness score:', results)
        print('True max: ', true_max)
        print('Number of maxima:', num_max)

    if plot:
        # Plot fitness scores
        plt.plot(fitness_scores)
        plt.show()

    return true_max, num_max


def plot_fitness_curves(prob, alg, curve):
    chart_title = 'Fitness Curve: ' + prob + ' - ' + alg
    plt.plot(curve)
    plt.title(chart_title, fontsize=18, y=1.03)
    plt.xlabel('Evaluations', fontsize=18)
    plt.ylabel('Fitness Score', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()


def generate_graph(num_nodes):

    # Generate high density graph (edges between all the nodes)
    all_possible_edges = []
    for n in range(num_nodes - 1):
        next_node = n + 1
        while next_node < num_nodes:
            all_possible_edges.append((n, next_node))
            next_node += 1

    # Generate graph based on target density
    density = 0.2  # Set to 1 for edges between all the nodes (incl. both (0, 1) and (1, 0)
    np.random.seed(42)
    edges = []
    for n in range(num_nodes):
        all_nodes = list(range(num_nodes))
        all_nodes.pop(n)  # no loops
        nodes_to_connect = np.random.choice(all_nodes, np.int(np.ceil(num_nodes * density)), replace=False)
        for e in nodes_to_connect:
            edges.append((n, e))

    print('number of nodes:', num_nodes)
    print('number of edges:', len(edges))
    print('edges:', edges)
    print()

    return edges
    # return all_possible_edges


def kcolors_max(state, edges):

    # Modify standard mlrose fitness function MaxKColor to evaluate the number of pairs of non-adjacent nodes of the
    # same color (instead of the number of pairs of adjacent nodes of the same color)
    # This is to turn this into a maximization problem, i.e. maximize the number of pairs of non-adjacent nodes of the
    # same color
    # Ref: https://mlrose.readthedocs.io/en/stable/_modules/mlrose/fitness.html#MaxKColor

    fitness_cnt = 0
    for i in range(len(edges)):
        # Check for non-adjacent nodes of the same color
        if state[edges[i][0]] != state[edges[i][1]]:  # '==' modified to '!='
            fitness_cnt += 1
    return fitness_cnt


def generate_knapsack(b, rs):

    w = []  # Initialize empty list for weights
    v = []  # # Initialize empty list for values
    np.random.seed(rs)
    for bit in range(b):
        w.append(np.random.randint(5, 15))
        v.append(np.random.randint(1, 5))
    return w, v


def fitness_function(f, bits, rs, verbose):

    if verbose:
        print('\n\n----------', f, ':', bits, 'bits ----------')

    if f == 'Four Peaks':
        fitness_fn = mlrose.FourPeaks(t_pct=0.15)  # Note: T= np.ceil(t_pct * n), per source code for FourPeaks.evaluate

    elif f == 'MaxKColor':
        # fitness_fn = mlrose.MaxKColor(edges)  # default mlrose fitness function
        # edges = [(0, 1), (0, 2), (1, 3), (2, 3)]  # 4 nodes, 2 by 2 grid, no diagonals
        # edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        edges = generate_graph(bits)
        kwargs = {'edges': edges}
        fitness_fn = mlrose.CustomFitness(kcolors_max, **kwargs)  # custom fitness function for maximization problem

    elif f == 'Knapsack':
        # weights = [10, 5, 2, 8, 15]
        # values = [1, 2, 3, 4, 5]
        weights, values = generate_knapsack(bits, rs)
        if verbose:
            print('\nKnapsack\n', weights, values)
        max_weight_pct = 0.6
        fitness_fn = mlrose.Knapsack(weights, values, max_weight_pct)

    elif f == 'FlipFlop':
        fitness_fn = mlrose.FlipFlop()

    # Check fitness for ad-hoc states
    # test_state = np.array([1, 0, 1, 1, 0])
    # print("Fitness for test_state", test_state, ":", fitness_fn.evaluate(test_state))

    return fitness_fn


def solve_problem(alg, prob, rs, b, attempts, verbose):

    if verbose:
        print('\n***', alg, '-', b, 'bits ***\n')

    t0 = time.time()

    if alg == 'Simulated Annealing':
        # Define decay schedule. https://mlrose.readthedocs.io/en/stable/source/decay.html
        # schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
        # schedule = mlrose.ArithDecay(init_temp=1.0, decay=0.0001, min_temp=0.001)
        # schedule = mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        schedule = mlrose.ExpDecay()
        # init_state = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Define initial state
        state, fit, curve = mlrose.simulated_annealing(prob, schedule=schedule, max_attempts=attempts, max_iters=np.inf,
                                                       init_state=None, curve=True, random_state=rs)

    elif alg == 'Genetic Algorithm':
        state, fit, curve = mlrose.genetic_alg(prob, pop_size=1000, mutation_prob=0.1, max_attempts=attempts,
                                               max_iters=np.inf, curve=True, random_state=rs)

    elif alg == 'Random Hill Climb':
        state, fit, curve = mlrose.random_hill_climb(prob, max_attempts=attempts, max_iters=np.inf, restarts=0,
                                                     init_state=None, curve=True, random_state=rs)

    elif alg == 'MIMIC':
        state, fit, curve = mlrose.mimic(prob, pop_size=500, keep_pct=0.2, max_attempts=attempts, max_iters=np.inf,
                                         curve=True, random_state=rs)

    else:
        state, fit, curve = 0, 0, 0

    t1 = time.time()
    runtime = t1-t0

    if verbose:
        # print("Best State:", state)
        print("Best Fitness:", fit)
        print("Number of fitness evaluations:", len(curve))
        np.savetxt("fitness_curve.csv", curve, delimiter=",")
        print('Solve problem run time: %0.3fs' % runtime)

    return state, fit, curve, runtime


def assignment_part_1():

    problems = ['Four Peaks', 'MaxKColor', 'Knapsack']
    algorithms = ['Simulated Annealing', 'Genetic Algorithm', 'Random Hill Climb', 'MIMIC']
    single = True  # True: Solve for single algorithm (e.g. validation, False: iterate through all
    rs = None  # Random state (use 'None' for none)
    verbose = True  # If true, print full solution space and corresponding fitness values
    plot = False
    all_results = []

    print('Start time:', time.strftime('%X %x %Z'))
    t_start = time.time()

    if single:  # Single algorithm for single bit size for a single problem (across multiple iterations)

        num_iterations = 1
        bits = 20  # Bit string length.  Note: only use <= 20 bits, evaluate_solution_space takes too long otherwise.
        attempts = [10]
        # attempts = [10, 20, 30, 40, 50]
        algorithm = algorithms[2]  # Select algorithm by changing algorithms list index
        prob = problems[2]  # Select problem by changing problems list index

        # Define fitness function
        fitness = fitness_function(prob, bits, rs, verbose)
        # Evaluate solution space for fitness function
        max_fitness, max_count = evaluate_solution_space(bits, fitness, plot, verbose)
        # Define problem (all discrete optimization, bit strings only)
        problem = mlrose.DiscreteOpt(length=bits, fitness_fn=fitness, maximize=True, max_val=2)

        # Repeat for different num_attempts
        for max_attempts in attempts:
            # Repeat across multiple iterations (with random state = None, results vary...)
            for iteration in range(num_iterations):
                reach_max = 0
                best_state, best_fitness, fitness_curve, run_time = solve_problem(algorithm, problem, rs, bits,
                                                                                  max_attempts, verbose)
                if best_fitness == max_fitness:
                    reach_max = 1
                all_results.append((bits, prob, algorithm, max_fitness, max_count, best_state, best_fitness,
                                    len(fitness_curve), run_time, reach_max, max_attempts, iteration))
                if plot:
                    plot_fitness_curves(prob, algorithm, fitness_curve)

        # Save results
        col = ('bits', 'prob', 'algorithm', 'max_fitness', 'max_count', 'best_state', 'best_fitness',
               'num_evaluations', 'run_time', 'reach_max', 'max_attempts', 'iteration')
        results_df = pd.DataFrame(all_results, columns=col)
        results_df.to_csv('results_df.csv')

    else:
        num_iterations = 5
        max_attempts = 10
        bit_range = [10, 20, 40, 70, 100]

        for prob in problems:  # Iterate through all the problems
            for bits in bit_range:  # Iterate through different bit string sizes
                # Define fitness function
                fitness = fitness_function(prob, bits, rs, verbose)
                # Evaluate solution space for fitness function
                # max_fitness, max_count = evaluate_solution_space(bits, fitness, plot, verbose)
                # Define problem (all discrete optimization, bit strings only)
                problem = mlrose.DiscreteOpt(length=bits, fitness_fn=fitness, maximize=True, max_val=2)
                for algorithm in algorithms:  # Iterating through all algorithms
                    for iteration in range(num_iterations):
                        # Solve problem
                        best_state, best_fitness, fitness_curve, run_time = solve_problem(algorithm, problem, rs, bits,
                                                                                          max_attempts, verbose)
                        all_results.append((bits, prob, algorithm, best_state, best_fitness,
                                            len(fitness_curve), run_time, iteration))
                        if plot:
                            plot_fitness_curves(prob, algorithm, fitness_curve)

        # Save results
        col = ('bits', 'prob', 'algorithm', 'best_state', 'best_fitness', 'num_evaluations',
               'run_time', 'iteration')
        results_df = pd.DataFrame(all_results, columns=col)
        results_df.to_csv('results_df_bits.csv')

        # # https://www.geeksforgeeks.org/iterating-over-rows-and-columns-in-pandas-dataframe/
        # for i, j in results_df.iterrows():
        #     print(i, j)
        #     print()

    print('\nEnd time:', time.strftime('%X %x %Z'))
    print('Done in %0.3fs' % (time.time() - t_start))


def mean_scores(train_scores, validation_scores, index_values):

    # Reused from Assignment 1 (rkaufholz3)

    # Calculate mean results from k-folds cross-validation
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    return train_scores_mean, validation_scores_mean


def plot_learning_curves(train_sizes, train_scores_mean, validation_scores_mean, chart_title):

    # Reused from Assignment 1 (rkaufholz3)

    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, validation_scores_mean, label='Validation score')
    plt.ylabel('F1 Score', fontsize=20)
    plt.xlabel('# Training Examples', fontsize=20)
    plt.title(chart_title, fontsize=24, y=1.03)
    # Ref: https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
    plt.legend(loc='best', prop={'size': 16})
    plt.ylim(0, 1.02)
    # Ref: https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.text(10, 0, text, fontsize=12)
    plt.grid()
    plt.show()


def manual_cross_validation(classifier, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                        stratify=y, shuffle=True)

    class_labels = ['0 <=50K', '1 >50K']

    # Check train and test data set sizes
    print('\nX_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    classifier.fit(X_train, y_train)

    # In sample accuracy (training score)
    y_pred_on_train = classifier.predict(X_train)
    print('\nTraining score - in sample accuracy')
    print("\nClassification report:\n", classification_report(y_train, y_pred_on_train, target_names=class_labels))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_on_train, labels=range(2)))
    print()
    print(f1_score(y_train, y_pred_on_train, average='weighted'))

    # Out of sample accuracy ('validation' / test score)
    y_pred = classifier.predict(X_test)
    print('\nValidation score - out of sample accuracy')
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_labels))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(2)))
    print()
    print(f1_score(y_test, y_pred, average='weighted'))


def learning_curves(X_train, y_train, alg):

    # Reused from Assignment 1 (rkaufholz3)

    # Ref: https://www.dataquest.io/blog/learning-curves-machine-learning/

    t0 = time.time()
    print('Learning curves time started:', time.strftime('%X %x %Z'))
    print('alg: ', alg)

    # Neural network with different optimization algorithms
    nodes = [100, 100]  # Number of nodes in each hidden layer.
    activation_fn = 'relu'  # 'identity', 'relu', 'sigmoid', 'tanh'
    num_restarts = 0  # Set for random hill climbing
    decay = mlrose.ExpDecay()  # Set for simulated annealing
    pop = 200  # Set for genetic algorithms
    mut_prob = 0.1  # Set for genetic algorithms
    rand_state = 42
    iters = 5000
    rate = 0.5
    attempts = 10
    classifier = mlrose.neural.NeuralNetwork(hidden_nodes=nodes, activation=activation_fn, algorithm=alg,
                                             max_iters=iters, bias=False, is_classifier=True, learning_rate=rate,
                                             early_stopping=False, restarts=num_restarts, schedule=decay, pop_size=pop,
                                             mutation_prob=mut_prob, max_attempts=attempts, random_state=rand_state)

    manual_cv = False

    if manual_cv:
        manual_cross_validation(classifier, X_train, y_train)
    else:
        # train_sizes = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 26170]
        # train_sizes = [20, 50, 100, 200, 500, 1000, 5000]  # 26170
        # train_sizes = [20, 50, 100]  # 26170
        train_sizes = [10, 200, 1000, 5000, 10000, 16362]
        # train_sizes = [10, 200]
        cv_value = 2
        chart_title = 'NN (' + alg + ')'

        # Set scoring metric
        scoring_metric = 'f1_weighted'

        # Get learning curves
        train_sizes, train_scores, validation_scores = learning_curve(classifier, X_train, y_train,
                                                                      train_sizes=train_sizes, cv=cv_value,
                                                                      shuffle=True,
                                                                      scoring=scoring_metric, n_jobs=-1)

        print('Learning curves time ended:', time.strftime('%X %x %Z'))
        print('Done in %0.3fs' % (time.time() - t0))
        print()

        # Calculate mean results from k-folds cross-validation
        train_scores_mean, validation_scores_mean = mean_scores(train_scores, validation_scores, train_sizes)

        # Plot learning curves
        params_text = "hidden_nodes=" + str(nodes) + ", activation=" + activation_fn + ", max_iters=" + str(iters) \
                      + "\nlearning_rate=" + str(rate) + ", max_attempts=" + str(attempts) + "\n"
        plot_learning_curves(train_sizes, train_scores_mean, validation_scores_mean, chart_title)

        print()
        print('train_scores_mean:', train_scores_mean)
        print('validation_scores_mean', validation_scores_mean)
        print()


def plot_validation_curves(train_scores_mean, validation_scores_mean, parameter, param_range, plot_type, chart_title):
    plt.title(chart_title, fontsize=24, y=1.03)
    plt.xlabel(parameter, fontsize=20)
    plt.ylabel("F1 Score", fontsize=20)
    plt.ylim(0.0, 1.02)

    # Ref: https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
    plt.tick_params(axis='both', which='major', labelsize=16)
    if plot_type == 'log_scale':
        plt.semilogx(param_range, train_scores_mean, label="Training score")
        plt.semilogx(param_range, validation_scores_mean, label="Validation score")
    if plot_type == 'linear_scale':
        plt.plot(param_range, train_scores_mean, label="Training score")
        plt.plot(param_range, validation_scores_mean, label="Validation score")
    # Ref: https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
    plt.legend(loc='best', prop={'size': 16})
    plt.grid()
    plt.show()


def validation_curves(X, y, alg):

    # Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    # Ref: https://scikit-learn.org/stable/modules/learning_curve.html

    t0 = time.time()
    print('Validation curves time started:', time.strftime('%X %x %Z'))
    print('X shape:', X.shape)
    print('Y shape:', y.shape)

    # Neural network with different optimization algorithms
    nodes = [100, 100]  # Number of nodes in each hidden layer.
    activation_fn = 'relu'  # 'identity', 'relu', 'sigmoid', 'tanh'
    num_restarts = 0  # Set for random hill climbing
    decay = mlrose.ExpDecay()  # Set for simulated annealing
    pop = 200  # Set for genetic algorithms
    mut_prob = 0.1  # Set for genetic algorithms
    iters = 2000
    rate = 0.1
    attempts = 10
    rand_state = 42
    classifier = mlrose.neural.NeuralNetwork(hidden_nodes=nodes, activation=activation_fn, algorithm=alg,
                                             max_iters=iters, bias=True, is_classifier=True, learning_rate=rate,
                                             early_stopping=False, restarts=num_restarts, schedule=decay, pop_size=pop,
                                             mutation_prob=mut_prob, max_attempts=attempts, random_state=rand_state)
    # parameter = 'max_iters'  # Parameter to be used in validation curve
    # param_range = (10, 50, 100, 150, 200, 500, 2000, 5000)
    # parameter = 'learning_rate'
    # param_range = (0.01, 0.05, 0.1, 0.2, 0.5)
    # parameter = 'pop_size'
    # param_range = (100, 200, 500, 1000)
    parameter = 'mutation_prob'
    param_range = (0.05, 0.1, 0.2, 0.3)
    cv_value = 5
    chart_title = 'NN (' + alg + ')'
    plot_type = 'linear_scale'

    # Set scoring metric
    # scoring_metric = 'accuracy'
    scoring_metric = 'f1_weighted'

    # Get validation curves
    train_scores, validation_scores = validation_curve(
        classifier, X, y, param_name=parameter, param_range=param_range,
        cv=cv_value, scoring=scoring_metric, n_jobs=-1)

    print('Validation curves time ended:', time.strftime('%X %x %Z'))
    print('Done in %0.3fs' % (time.time() - t0))
    print()

    # Calculate mean results from k-folds cross-validation
    train_scores_mean, validation_scores_mean = mean_scores(train_scores, validation_scores, param_range)

    # Plot validation curves
    plot_validation_curves(train_scores_mean, validation_scores_mean, parameter, param_range, plot_type, chart_title)


def prediction(X_train, y_train, X_test, y_test, alg):

    # Modified from Assignment 1 (rkaufholz3)

    # Note: insert relevant optimal parameters based on learning and validation curve analysis
    # Note: manually iterate for increase training_size size to generate Test learning curve.

    # class_labels = ['0 T-shirt/top', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt',
    #                 '7 Sneaker', '8 Bag', '9 Ankle boot']

    print()
    print('---------- alg: ', alg, '-----------')
    print()

    class_labels = ['0 <=50K', '1 >50K']

    # Check train and test data set sizes
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # Neural network with different optimization algorithms
    nodes = [100, 100]  # Number of nodes in each hidden layer.
    activation_fn = 'relu'  # 'identity', 'relu', 'sigmoid', 'tanh'
    num_restarts = 0  # Set for random hill climbing
    decay = mlrose.ExpDecay()  # Set for simulated annealing
    pop = 200  # Set for genetic algorithms
    mut_prob = 0.1  # Set for genetic algorithms
    rand_state = 42
    iters = 5000
    rate = 0.5
    attempts = 10
    classifier = mlrose.neural.NeuralNetwork(hidden_nodes=nodes, activation=activation_fn, algorithm=alg,
                                             max_iters=iters, bias=False, is_classifier=True, learning_rate=rate,
                                             early_stopping=False, restarts=num_restarts, schedule=decay, pop_size=pop,
                                             mutation_prob=mut_prob, max_attempts=attempts, random_state=rand_state)

    # Fit
    t0 = time.time()
    print('\nTime fit started:', time.strftime('%X %x %Z'))
    classifier.fit(X_train, y_train)
    print('Time fit ended:', time.strftime('%X %x %Z'))
    print('Fit done in %0.3fs' % (time.time() - t0))

    # Predict
    t0 = time.time()
    print('\nTime predict started:', time.strftime('%X %x %Z'))
    y_pred = classifier.predict(X_test)
    print('Time predict ended:', time.strftime('%X %x %Z'))
    print('Predict done in %0.3fs' % (time.time() - t0))

    # Print results
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_labels))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(2)))


def assignment_part_2():

    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
    algorithm = algorithms[0]  # Select problem by changing problems list index

    # Load and pre-process Adult data
    training_size = 0  # Set to 0 for Learning Curves
    X_train, X_test, y_train, y_test = load_adult_data(training_size)

    # Learning curves
    # learning_curves(X_train, y_train, algorithm)

    # Validation curves
    # validation_curves(X_train, y_train, algorithm)

    # Prediction
    prediction(X_train, y_train, X_test, y_test, algorithm)


if __name__ == "__main__":

    # Choose assignment part
    part = 2  # 1=Random Optimization Problems, 2=Neural Network Training
    if part == 1:
        assignment_part_1()
    if part == 2:
        assignment_part_2()
