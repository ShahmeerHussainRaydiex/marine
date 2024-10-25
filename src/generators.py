# Imports
import copy
import itertools
import random


# Functions
def flatten(nested_list):
    for element in nested_list:
        if isinstance(element, list):
            yield from flatten(element)
        else:
            yield element


def metric_generator(method=None, base_dict=None, techs=None, metrics=None, N=2, step=0.1, num=1_000):
    """Generates a list of technology dicts with adjustment factors based on the user inputs."""

    # create list of factors to test
    factors = [i * step for i in range(0, N + 1)]

    # Add the negative factors
    factors += [-i for i in factors]

    # create list of options to test for each tech for each metric for each factor (params)
    params = []
    for i, techs in enumerate(techs):
        for j, metric in enumerate(metrics):
            fl = []
            for k, factor in enumerate(factors):
                fl.append((techs, metric, factor))
            params.append(fl)
    # generate combinations based on selected method
    if method == "single-factor":
        combinations = list(flatten(params))

    elif method == "cartesian":
        combinations = list(itertools.product(*params))

    elif method == "monte-carlo":
        # print(params)
        x = list(itertools.product(*params))
        combinations = []
        for _ in range(num):
            combinations.append(random.choice(x))

    # apply generated combinations to test into dictionaries
    params_to_simulate = {}
    for idx, option in enumerate(combinations):
        d = copy.deepcopy(base_dict)
        n = f"{method}-{idx}"
        for item in option:
            tech, metric, factor = item

            d[tech]['metric_mods'][metric] = 1 + factor

        params_to_simulate[n] = d

    return params_to_simulate


def toy_generator(base_dict=None, count=10):

    instances_to_run = {}

    for i in range(count):  # just a dummy loop to test compute times...
        config = copy.deepcopy(base_dict)
        name = f"option_{i}"
        instances_to_run[name] = config

    return instances_to_run


# unused...
def SFST(base_dict=None, tech=None, metrics=None, N=None, step=None):
    """Simple function to generate single-factor sensitivity for one metric(factor) and
    one technology across N*2 options with step size of step. If you pass in a list of
    metrics, each one will have the same factors applied to each."""

    # N = 3
    # step = 0.05
    # techs = 'monopile'
    # metrics = ['cap', 'opr', 'energy', 'val']

    factors = [i * step for i in range(-N, N + 1)]

    adjustments = []

    # cycle over options
    for metric in metrics:
        for factor in factors:
            d = copy.deepcopy(base_single_techs)
            d[tech] = {metric: factor}
            adjustments.append(d)

    return adjustments






def goals_generator(goals={}, N=2, step=0.1, num=1_000):
    ''' 
    Generates random goals based on the input goals dictionary and the steps and number of goals to generate.

    Does not generate every possible combination but just randomly modifies the values... could be updated to be more
    similar to the metric_generator function.
    '''

    goals_to_simulate = {}

    factors = [i * step for i in range(0, N + 1)]

    for i in range(num):
        temp_goals = copy.deepcopy(goals)

        for goal, values in temp_goals.items():
            goal_value = (list(values.values())[0])

            # Multiply it by a random factor
            factor = random.choice(factors)
            goal_value = goal_value * (1 + factor)

            # Update the goals with the new value as well as the other values
            temp_goals[goal][list(values.keys())[0]] = goal_value

        goals_to_simulate[f'monte-carlo-{i}'] = temp_goals

    return goals_to_simulate
