import pandas as pd
from matplotlib import pyplot as plt

results = pd.read_csv('results.csv', index_col=0)
england = results[results['winner'] == 'England'].reset_index()

run_wins = england[england['win_mechanism'] == 'runs']
wicket_wins = england[england['win_mechanism'] == 'wickets']

plt.scatter(run_wins['match_id'], run_wins['score'])
plt.scatter(wicket_wins['match_id'], wicket_wins['score'])
plt.show()