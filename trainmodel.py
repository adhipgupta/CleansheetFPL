import argparse
import pandas as pd
from fbrefscraper import create_final_defender_table, get_fixtures, merge_attack_defense, get_attack_table, get_defense_table
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Integer

url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
#Preprocess and create defender and attack team data tables
fbref_pl_data = pd.read_html(url)
defense_table = get_defense_table(fbref_pl_data)
attack_table = get_attack_table(fbref_pl_data)
defender_data = create_final_defender_table(fbref_pl_data)
defender_numeric = defender_data.drop([ 'team', 'opponent_team_name'], axis=1)



def train_model(X, n_estimators=350, max_depth=5, learning_rate=0.2, gamma=0.5, min_child_weight=6):
    #Train the Model
    y = X.clean_sheets
    X.drop('clean_sheets', axis=1, inplace=True)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,
                                                                    random_state=0, shuffle=False)
    my_model_1 = XGBRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                              gamma=gamma)

    my_model_1.fit(X_train_full, y_train) # Your code here
    predictions_1 =  my_model_1.predict(X_valid_full)
    mae = mean_absolute_error(predictions_1, y_valid) # Your code here
    return my_model_1, mae

model, mae = train_model(defender_numeric)
print (mae)
# save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

def parameter_tuning():

    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.01, 0.1, 0.15, 0.2],
        'max_depth': [4, 5],
        'gamma': [0, 0.1, 0.5, 0.7],
        #'min_child_weight': [1, 5, 10],
        #'subsample': [0.5, 0.4, 0.7, 1],
        #'colsample_bytree': [0.5, 0.7, 1],
        #'reg_alpha': [0, 0.01, 0.05, 0.1],
        #'reg_lambda': [0, 0.01, 0.05, 0.1]
    }
    xgb = XGBRegressor()

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5)
    y = defender_numeric.clean_sheets
    X = defender_numeric.drop('clean_sheets', axis=1)
    grid_search.fit(X, y)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    return grid_search.best_params_
