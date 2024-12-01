# CricketPredictor
Predicts the final score for a men's international T20 cricket match.

## How it works
The model uses least squares regression to predict the runs scored by each team, based on 1026 historical fixtures.
Teams are ranked by total runs scored across the dataset in ascending order. The rank of a team is the numerical value assigned to it.

The inputs to the model are:
- Team A rank
- Team B rank

The outputs are:
- Team A total expected runs
- Team B total expected runs

### To predict a specific fixture

```
pip install -r requirements.txt
python predict_single.py TEAM_A_ID TEAM_B_ID
```

Team IDs can be found in 'team_map.csv'.

## Performance
Using the L2 norm of the error, averaged over the train/test dataset, the following errors are observed.

- Train error ~ 0.006
- Test error ~ 0.01

## Potential improvements
- The input data is fairly low dimensional, therefore delay embedding could be used in order to reconstruct a higher dimensional state space.
- Once data is delay embedded, adding nonlinearity to the model (as demonstrated by the train_nonlinear function in train.py) may reduce both test and train errors.
- An alternative to delay embedding would be to find additional dimensions of the full dataset which could be used  as inputs.
- If nonlinear regression is still resulting in unsatisfactory performance, an ESN could be used.