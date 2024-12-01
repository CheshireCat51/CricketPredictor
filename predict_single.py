from train import train, get_unique_teams, sort_teams, max_runs
import sys
import numpy as np


def predict_single(XX: np.ndarray, team_ids: list):
   
    WW = train()
    pred = WW @ XX
    translate(pred, team_ids)


def translate(pred, team_ids):

    unique_teams = get_unique_teams()

    team_a_score = round(pred[0]*max_runs, 3)
    team_b_score = round(pred[1]*max_runs, 3)

    print(unique_teams[team_ids[0]-1], team_a_score, '-', team_b_score, unique_teams[team_ids[1]-1])


if __name__ == '__main__':
    team_a_id = int(sys.argv[1])
    team_b_id = int(sys.argv[2])

    if team_a_id < 1 or team_a_id > 84:
        print('Invalid Team ID for team A. Please try again.')
        sys.exit()

    if team_b_id < 1 or team_b_id > 84:
        print('Invalid Team ID for team B. Please try again.')
        sys.exit()

    unique_teams = get_unique_teams()
    sorted_teams = sort_teams()
    team_a_rank = [i+1 for i in range(len(sorted_teams)) if unique_teams[team_a_id-1] == sorted_teams[i]][0]
    team_b_rank = [i+1 for i in range(len(sorted_teams)) if unique_teams[team_b_id-1] == sorted_teams[i]][0]

    print(team_a_rank)
    print(team_b_rank)

    predict_single(np.array([team_a_rank, team_b_rank]), [team_a_id, team_b_id])