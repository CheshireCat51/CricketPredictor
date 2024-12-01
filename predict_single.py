from train import train, get_unique_teams, max_runs
import sys
import numpy as np


def predict_single(XX: np.ndarray):
   
    WW = train()
    pred = WW @ XX
    translate(pred, XX)


def translate(pred, input):

    unique_teams = get_unique_teams()

    team_a_score = round(pred[0]*max_runs, 3)
    team_b_score = round(pred[1]*max_runs, 3)

    print(unique_teams[input[0]-1], team_a_score, '-', team_b_score, unique_teams[input[1]-1])


if __name__ == '__main__':
    team_a_id = int(sys.argv[1])
    team_b_id = int(sys.argv[2])

    if team_a_id < 0 or team_a_id > 84:
        print('Invalid Team ID for team A. Please try again.')
        sys.exit()

    if team_b_id < 0 or team_b_id > 84:
        print('Invalid Team ID for team B. Please try again.')
        sys.exit()

    predict_single(np.array([team_a_id, team_b_id]))