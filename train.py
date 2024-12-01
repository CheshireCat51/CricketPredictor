import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import random
import pandas as pd
from itertools import combinations_with_replacement


results = pd.read_csv('results.csv', index_col=0)
max_runs = max(results[['team_a_runs', 'team_b_runs']].max().values)


def get_unique_teams():

    unique_teams = sorted(list(set(results['team_a'].tolist() + results['team_b'].tolist())))

    return unique_teams


def sort_teams():

    """Sort teams by total runs scored as a measure of team strength."""

    unique_teams = get_unique_teams()

    run_sums = []
    for team in unique_teams:
        run_sum = 0
        for index, row in results.iterrows():
            if row['team_a'] == team:
                run_sum += row['team_a_runs']
            elif row['team_b'] == team:
                run_sum += row['team_b_runs']
        run_sums.append(run_sum)

    sorted_run_sums = np.argsort(run_sums)
    sorted_teams = np.array(unique_teams)[sorted_run_sums]  # Sort teams by runs scored

    return sorted_teams


def wrangling():
   
    sorted_teams = sort_teams()
    wrangled_results = results.replace(sorted_teams, list(range(1, len(sorted_teams)+1)))
    # wrangled_results['team_a'] /= len(sorted_teams)
    # wrangled_results['team_b'] /= len(sorted_teams)
    wrangled_results['team_a_runs'] /= max_runs
    wrangled_results['team_b_runs'] /= max_runs

    return wrangled_results


def train_test_split(XX: np.ndarray, YY: np.ndarray, train_proportion: float):

    train_ids = sorted(random.sample(range(XX.shape[1]), round(train_proportion*XX.shape[1])))
    test_ids = list(set(range(XX.shape[1])).difference(train_ids))

    XX_train = XX[:,train_ids]
    XX_test = XX[:,test_ids]
    YY_train = YY[:,train_ids]
    YY_test = YY[:,test_ids]

    return XX_train, XX_test, YY_train, YY_test


def train():

    wrangled_results = wrangling()

    # Split into train and test
    XX_train, XX_test, YY_train, YY_test = train_test_split(wrangled_results[['team_a', 'team_b']].values.T, wrangled_results[['team_a_runs', 'team_b_runs']].values.T, 0.8)

    # Create model
    WW = (YY_train @ XX_train.T) @ la.pinv(XX_train @ XX_train.T)

    err_train = np.mean(la.norm(WW @ XX_train - YY_train, axis=0, ord=2) / np.sqrt(YY_train.shape[1]))
    err_test = np.mean(la.norm(WW @ XX_test - YY_test, axis=0, ord=2) / np.sqrt(YY_test.shape[1]))

    print('Train err:', err_train)
    print('Test err:', err_test)

    # predict(WW, XX_test, YY_test)

    return WW


def predict(WW: np.ndarray, XX: np.ndarray, YY: np.ndarray):
   
    pred = WW @ XX

    for i in range(XX.shape[1]):
        translate(YY[:,i], pred[:,i], XX[:,i])
        print('\n')


def translate(truth, pred, input):

    unique_teams = sort_teams()

    for i in [truth, pred]:
        team_a_score = round(i[0]*max_runs, 3)
        team_b_score = round(i[1]*max_runs, 3)
        # team_a_id = int(input[0]*len(unique_teams))-1
        # team_b_id = int(input[1]*len(unique_teams))-1
        print(unique_teams[input[0]-1], team_a_score, '-', team_b_score, unique_teams[input[1]-1])


def train_nonlinear():

    """WIP."""

    def lib_of_functions(XX, min_order, max_order):
        X_nonlin = []
        if min_order == 0:
            X_nonlin.append(np.ones(XX.shape[1]))

        for n in range(max(1, min_order), max_order+1):
            ll = list(combinations_with_replacement(range(XX.shape[0]), n))
            for k in ll:
                X_nonlin.append(XX[k,:].prod(0))

        XX_hat = np.vstack(X_nonlin)

        return XX_hat

    wrangled_results = wrangling()
    
    # Split into train and test
    XX_train, XX_test, YY_train, YY_test = train_test_split(wrangled_results[['team_a', 'team_b']].values.T, wrangled_results[['team_a_runs', 'team_b_runs']].values.T, 0.8)

    CC = XX_train @ XX_train.T  # Create correlation matrix
    U, S, Vt = la.svd(CC, hermitian=True)  # SVD
    rank = 2

    train_errs = []
    test_errs = []

    for order in [1, 2, 3]:

        XX_train_reduced = U[:,:rank].T @ XX_train
        YY_train_reduced = U[:,:rank].T @ YY_train
        XX_train_reduced_hat = lib_of_functions(XX_train_reduced, 0, order)

        XX_test_reduced = U[:,:rank].T @ XX_test
        YY_test_reduced = U[:,:rank].T @ YY_test
        XX_test_reduced_hat = lib_of_functions(XX_test_reduced, 0, order)

        WW = (YY_train_reduced @ XX_train_reduced_hat.T) @ la.pinv(XX_train_reduced_hat @ XX_train_reduced_hat.T)

        err_train = la.norm(WW @ XX_train_reduced_hat - YY_train_reduced, ord=2) / YY_train_reduced.shape[1]
        err_test = la.norm(WW @ XX_test_reduced_hat - YY_test_reduced, ord=2) / YY_test_reduced.shape[1]

        train_errs.append(err_train)
        test_errs.append(err_test)

        print(f'{order} order')
        print('Train err:', err_train)
        print('Test err:', err_test)
        print('\n')

    return WW


if __name__ == '__main__':
    train()
    # train_nonlinear()