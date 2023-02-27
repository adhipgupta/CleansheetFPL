import argparse
import pandas as pd
from fbrefscraper import create_final_defender_table, get_fixtures, merge_attack_defense, get_attack_table, get_defense_table
import pickle

url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
#Preprocess and create defender and attack team data tables
fbref_pl_data = pd.read_html(url)
defense_table = get_defense_table(fbref_pl_data)
attack_table = get_attack_table(fbref_pl_data)
defender_data = create_final_defender_table(fbref_pl_data)
defender_numeric = defender_data.drop([ 'team', 'opponent_team_name'], axis=1)
model, mae = train_model(defender_numeric)

# save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

def train_model(X, n_estimators=300, max_depth=5, learning_rate=0.15):
    #Train the Model
    y = X.clean_sheets
    X.drop('clean_sheets', axis=1, inplace=True)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,
                                                                    random_state=0, shuffle=False)
    my_model_1 = XGBRegressor(random_state=0, n_estimators=300, max_depth=5, learning_rate=0.15)


    my_model_1.fit(X_train_full, y_train) # Your code here
    predictions_1 =  my_model_1.predict(X_valid_full)
    mae = mean_absolute_error(predictions_1, y_valid) # Your code here
    return my_model_1, mae
