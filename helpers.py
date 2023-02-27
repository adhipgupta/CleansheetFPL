import pandas as pd
import pickle
from fbrefscraper import get_fixtures, get_attack_table, get_defense_table

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fbref advanced PL team data
url = 'https://fbref.com/en/comps/9/Premier-League-Stats'

#Preprocess and create defender and attack team data tables
fbref_pl_data = pd.read_html(url)
defense_table = get_defense_table(fbref_pl_data)
attack_table = get_attack_table(fbref_pl_data)

def get_fixtures_gw(gw):
    gw = int(gw)
    gws_all_fixtures = get_fixtures()
    gameweek_fixtures = gws_all_fixtures[gws_all_fixtures['GW'] == gw]
    print (gw)
    return gameweek_fixtures

def get_all_data_for_gw(gw):
    gw_table = get_fixtures_gw(gw)

    #Merge Attack and defense table to the GW table
    gw_table = pd.merge(gw_table, defense_table, left_on='team', right_on='Squad', how='left')
    gw_table = pd.merge(gw_table, attack_table, left_on='opponent_team_name', right_on='Squad', how='left')
    gw_table.drop('Squad_x', axis=1, inplace=True)
    gw_table.drop('Squad_y', axis=1, inplace=True)

    #gw_table = merge_attack_defense(defender_data, attack_table, gw_table)
    return gw_table

def predict_clean_sheet(gw):
    gw_fixtures = get_all_data_for_gw(gw)
    X_full = gw_fixtures.drop([ 'team', 'opponent_team_name'], axis=1)
    X_full.fillna(0, inplace=True)
    predictions_1 =  model.predict(X_full)
    return gw_fixtures, predictions_1

def print_predictions(gw_fixtures, predictions):
    print("{:<20} {:<20} {:<20}".format('CleanSheet', 'Team', 'Opposition'))
    k = 0
    for val in predictions:
        print ("{:<20} {:<20} {:<20}".format(val, gw_fixtures.team[k], gw_fixtures.opponent_team_name[k]))
        k = k + 1