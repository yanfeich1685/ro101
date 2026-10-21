import time
from gurobipy import GRB
import gurobipy as gp
import numpy as np


c = np.array([3, 4, 5, 6])
a = np.array([0.4, 0.5, 0.8, 0.9])
d = np.array([0.1, 0.1, 0.1, 0.05])
b = 100


def deterministic():
    print('=' * 80)
    print('Deterministic production problem')
    print('=' * 80)

    tic = time.perf_counter()
    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addMVar(shape=4, lb=0, vtype=GRB.CONTINUOUS)  # matrix variable

    model.setObjective(c @ x, GRB.MINIMIZE) # matrix product

    model.addConstr(a @ x >= b)

    model.optimize()

    obj = round(model.ObjVal, 2)
    sol = x.X.round(2)
    elapsed = time.perf_counter() - tic

    print(f'Runtime = {elapsed:.2f}')
    print(f'Minimum cost = {obj}')
    print('Optimal solution:')
    for i in range(4):
        if sol[i] > 1e-6:
            print(f'x_{i + 1} = {sol[i]}') # +1, because in Python the index starts from 0
    print()


def budgeted():
    print('=' * 80)
    print('Production problem under budgeted uncertainty')
    print('Robust counterpart via duality')
    print('=' * 80)

    for Gamma in range(5):
        tic = time.perf_counter()
        model = gp.Model()
        model.Params.OutputFlag = 0

        x = model.addMVar(shape=4, lb=0, vtype=GRB.CONTINUOUS)
        q = model.addMVar(shape=4, lb=0, vtype=GRB.CONTINUOUS)
        z = model.addVar(lb=0, vtype=GRB.CONTINUOUS)

        model.setObjective(c @ x, GRB.MINIMIZE)

        model.addConstr(a @ x - Gamma * z - q.sum() >= b)
        model.addConstr(q >= d * x - z)

        model.optimize()

        obj = round(model.ObjVal, 2)
        sol = x.X.round(2)
        elapsed = time.perf_counter() - tic

        if Gamma == 4:
            print(f'Gamma = {Gamma} (box)')
        elif Gamma == 0:
            print(f'Gamma = {Gamma} (deterministic)')
        else:
            print(f'Gamma = {Gamma}')
        print(f'Runtime = {elapsed:.2f}')
        print(f'Minimum cost = {obj}')
        print('Optimal solution:')
        for i in range(4):
            if sol[i] > 1e-6:
                print(f'x_{i + 1} = {sol[i]}')
        print()


def simulation(Gamma, sce):
    print('=' * 80)
    print(f'Production problem under budgeted uncertainty, Gamma = {Gamma}')
    print('Simulation')
    print('=' * 80)

    n_sim_list = [10, 50, 100, 500, 1000] # number of scenarios
    for n_sim in n_sim_list:
        tic = time.perf_counter()
        model = gp.Model()
        model.Params.OutputFlag = 0

        x = model.addMVar(shape=4, lb=0, vtype=GRB.CONTINUOUS)

        model.setObjective(c @ x, GRB.MINIMIZE)

        model.addConstr(a @ x >= b)

        for s in range(n_sim):
            xi = sce[s]
            xi = xi / xi.sum() * Gamma # normalization
            a_new = a - xi * d
            model.addConstr(a_new @ x >= b)

        model.optimize()

        obj = round(model.ObjVal, 2)
        sol = x.X.round(2)
        elapsed = time.perf_counter() - tic

        print(f'Number of simulations = {n_sim}')
        print(f'Runtime = {elapsed:.2f}')
        print(f'Minimum cost = {obj}')
        print('Optimal solution:')
        for i in range(4):
            if sol[i] > 1e-6:
                print(f'x_{i + 1} = {sol[i]}')
        print()


def vertices():
    print('=' * 80)
    print('Production problem under budgeted uncertainty, Gamma = 1')
    print('Vertex enumeration')
    print('=' * 80)

    tic = time.perf_counter()

    v2 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, 0])
    v4 = np.array([0, 0, 1, 0])
    v5 = np.array([0, 0, 0, 1])

    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addMVar(shape=4, lb=0, vtype=GRB.CONTINUOUS)

    model.setObjective(c @ x, GRB.MINIMIZE)

    model.addConstr(a @ x >= b)

    model.addConstr(a @ x - v2 * d * x >= b)
    model.addConstr(a @ x - v3 * d * x >= b)
    model.addConstr(a @ x - v4 * d * x >= b)
    model.addConstr(a @ x - v5 * d * x >= b)

    model.optimize()

    obj = round(model.ObjVal, 2)
    sol = x.X.round(2)
    elapsed = time.perf_counter() - tic

    print(f'Runtime = {elapsed:.2f}')
    print(f'Minimum cost = {obj}')
    print('Optimal solution:')
    for i in range(4):
        if sol[i] > 1e-6:
            print(f'x_{i + 1} = {sol[i]}')
    print()


if __name__ == '__main__':   
    deterministic()
    budgeted()
    
    '''
    We sample 1000 scenarios and store them in sce.
    In this case, the scenario sets are nested for #samples = 10, 50, 100, 500, 1000.
    This guarantees monotonicity of the objective value.
    '''
    np.random.seed(1077)    
    sce = np.random.rand(1000, 4)
    simulation(1, sce)
    
    vertices()
