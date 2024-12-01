import pandas as pd
import numpy as np
import json
import glob


deliveries = []
results = []

for file_name in glob.glob('./all_male_json/*.json'):
    with open(file_name, 'r') as file:
        match_data = json.load(file)

    if match_data['info']['team_type'] == 'international' and match_data['info']['gender']  == 'male':
        match_id = file_name.split('\\')[1].replace('.json', '')
        print(match_id)

        outcome = match_data['info']['outcome']
        try:
            winner = outcome['winner']
        except KeyError:
            continue

        team_a = match_data['innings'][0]['team']
        team_a_runs = 0
        for i in match_data['innings'][0]['overs']:
            for j in i['deliveries']:
                team_a_runs += j['runs']['total']
        
        team_b = match_data['innings'][1]['team']
        team_b_runs = 0
        for i in match_data['innings'][1]['overs']:
            for j in i['deliveries']:
                team_b_runs += j['runs']['total']

        row = {
            'match_id': match_id,
            'team_a': team_a,
            'team_b': team_b,
            'team_a_runs': team_a_runs,
            'team_b_runs': team_b_runs
        }
        results.append(row)

        for inning in match_data['innings']:
            for over in inning['overs']:
                for deliv in over['deliveries']:
                    wicket = False
                    if 'wickets' in deliv.keys():
                        wicket = True
                    row = {
                        'match_id': match_id,
                        'over_id': over['over'],
                        'bowler': deliv['bowler'],
                        'batter': deliv['batter'],
                        'non_striker': deliv['non_striker'],
                        'runs': deliv['runs']['total'],
                        'wicket': wicket
                    }
                    deliveries.append(row)

deliveries = pd.DataFrame(deliveries)
deliveries.to_csv('deliveries_all.csv')
results = pd.DataFrame(results)
results.to_csv('results_all.csv')