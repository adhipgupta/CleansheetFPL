import argparse
import pandas as pd
from fbrefscraper import create_final_defender_table, get_fixtures, merge_attack_defense, get_attack_table, get_defense_table, train_model
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

