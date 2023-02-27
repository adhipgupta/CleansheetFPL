import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


id_to_team = {
    16: 'Nott\'m Forest',
    4:	'Brentford',
    9:	'Fulham',
    12:	'Liverpool',
    17:	'Southampton',
    1:	'Arsenal',
    18:	'Spurs',
    15:	'Newcastle',
    3:	'Bournemouth',
    19:	'West Ham',
    2:	'Aston Villa',
    10:	'Leicester',
    13:	'Man City',
    6:	'Chelsea',
    14:	'Man Utd',
    8:	'Everton',
    7:	'Crystal Palace',
    11:	'Leeds',
    5:	'Brighton',
    20:	'Wolves'}

def get_defense_table(df):
    goalkeeping_defense = df[4]
    selected_columns = goalkeeping_defense['Unnamed: 0_level_0'][['Squad']]
    defense_table = pd.DataFrame(selected_columns)
    defense_table[['SoTA', 'Saves', 'Save%']] = goalkeeping_defense['Performance'].loc[:, ['SoTA', 'Saves', 'Save%']]
   
    # Get Advanced Goalkeeping Stats
    adv_goalkeeping = df[6]
    defense_table[['CK']] = adv_goalkeeping['Goals'].loc[:,['CK']]
    defense_table[['Cmp', 'Att']] = adv_goalkeeping['Launched'].loc[:,['Cmp', 'Att']]
    defense_table[['PassAtt', 'PassLaunch%', 'AvgLen']] = adv_goalkeeping['Passes'].loc[:,['Att', 'Launch%', 'AvgLen']]
    defense_table[['Opp', 'Stp%']] = adv_goalkeeping['Crosses'].loc[:,['Opp', 'Stp']]
    defense_table[['OPA/90', 'AvGDist']] = adv_goalkeeping['Sweeper'].loc[:,['#OPA/90','AvgDist']]

    defensive_actions = df[16]
    defense_table[['Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd']] = defensive_actions['Tackles'].loc[:,['Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd']]
    defensive_actions = df[16]
    defense_table[['CTkl', 'CAtt', 'Lost']] = defensive_actions['Challenges'].loc[:,['Tkl', 'Att', 'Lost']]
    defense_table[['Blocks']] = defensive_actions['Blocks'].loc[:,['Blocks']]
    defense_table[['Blocks']] = defensive_actions['Blocks'].loc[:,['Blocks']]
    defense_table[['Int']] = defensive_actions['Unnamed: 15_level_0'].loc[:,['Int']]
    defense_table[['Tkl+Int']] = defensive_actions['Unnamed: 16_level_0'].loc[:,['Tkl+Int']]
    defense_table[['Clr']] = defensive_actions['Unnamed: 17_level_0'].loc[:,['Clr']]
    defense_table[['Err']] = defensive_actions['Unnamed: 18_level_0'].loc[:,['Err']]

    passing_opp = df[11]

    defense_table[['KP']] = passing_opp['Unnamed: 21_level_0'].loc[:,['KP']]
    defense_table[['1/3']] = passing_opp['Unnamed: 22_level_0'].loc[:,['1/3']]
    defense_table[['PPA']] = passing_opp['Unnamed: 23_level_0'].loc[:,['PPA']]
    defense_table[['CrsPA']] = passing_opp['Unnamed: 24_level_0'].loc[:,['CrsPA']]
    defense_table[['PrgP']] = passing_opp['Unnamed: 25_level_0'].loc[:,['PrgP']]
    return defense_table

def get_attack_table(df):
    shooting = df[8]
    selected_columns = shooting['Unnamed: 0_level_0'][['Squad']]
    attack_table = pd.DataFrame(selected_columns)

    attack_table[['Sh', 'SoT', 'Dist', 'FK']] = shooting['Standard'].loc[:,['Sh', 'SoT', 'Dist', 'FK']]
    attack_table[['xG']] = shooting['Expected'].loc[:,['xG']]


    passing = df[10]
    attack_table[['Cmp', 'Att', 'TotDist', 'PrgDist']] = passing['Total'].loc[:,['Cmp', 'Att', 'TotDist', 'PrgDist']]
    attack_table[['SCmp', 'SAtt']] = passing['Short'].loc[:,['Cmp', 'Att']]
    attack_table[['MCmp', 'MAtt']] = passing['Medium'].loc[:,['Cmp', 'Att']]
    attack_table[['LCmp', 'LAtt']] = passing['Long'].loc[:,['Cmp', 'Att']]
    attack_table[['KP']] = passing['Unnamed: 21_level_0'].loc[:,['KP']]
    attack_table[['1/3']] = passing['Unnamed: 22_level_0'].loc[:,['1/3']]
    attack_table[['PPA']] = passing['Unnamed: 23_level_0'].loc[:,['PPA']]
    attack_table[['CrsPA']] = passing['Unnamed: 24_level_0'].loc[:,['CrsPA']]
    attack_table[['PrgP']] = passing['Unnamed: 25_level_0'].loc[:,['PrgP']]

    passing_type = df[12]
    attack_table[['Live', 'Dead', 'FK', 'TB', 'Sw', 'Crs', 'TI', 'CK']] = passing_type['Pass Types'].loc[:,['Live', 'Dead', 'FK', 'TB', 'Sw', 'Crs', 'TI', 'CK']]

    possession = df[18]
    attack_table[['poss']] = possession['Unnamed: 2_level_0'].loc[:,['Poss']]
    attack_table[['Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Live']] = possession['Touches'].loc[:,['Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Live']]
    attack_table[['Carries', 'TotDist', 'PrgDist', 'PrgC', '1/3', 'CPA', 'Mis', 'Dis']] = possession['Carries'].loc[:,['Carries', 'TotDist', 'PrgDist', 'PrgC', '1/3', 'CPA', 'Mis', 'Dis']]

    return attack_table

def create_final_defender_table(df):
    defense_table = get_defense_table(df)
    attack_table = get_attack_table(df)
    
    # Access the Latest GW data
    merged_gw_data_url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv'
    final_table = pd.read_csv(merged_gw_data_url)

    #Remove Fwd and mids. Remove players with minutes = 0.
    final_table = final_table.loc[final_table['position'] != 'FWD']
    final_table = final_table.loc[final_table['position'] != 'MID']
    final_table = final_table.loc[final_table['minutes'] != 0]
    final_table = final_table.loc[:,['name', 'team', 'position', 'clean_sheets', 'GW', 'opponent_team', 'was_home']]

    #Remove duplicates
    final_table.drop_duplicates(subset=['team', 'GW'], inplace=True)

    #Rename the Squad names to match github and fbref data
    defense_table['Squad'] = defense_table['Squad'].replace('Leicester City', 'Leicester')
    defense_table['Squad'] = defense_table['Squad'].replace('Manchester City', 'Man City')
    defense_table['Squad'] = defense_table['Squad'].replace('Manchester Utd', 'Man Utd')
    defense_table['Squad'] = defense_table['Squad'].replace('Tottenham', 'Spurs')
    defense_table['Squad'] = defense_table['Squad'].replace('Nott\'ham Forest', 'Nott\'m Forest')
    defense_table['Squad'] = defense_table['Squad'].replace('Newcastle Utd', 'Newcastle')
    defense_table['Squad'] = defense_table['Squad'].replace('Leeds United', 'Leeds')

    attack_table['Squad'] = attack_table['Squad'].replace('Leicester City', 'Leicester')
    attack_table['Squad'] = attack_table['Squad'].replace('Manchester City', 'Man City')
    attack_table['Squad'] = attack_table['Squad'].replace('Manchester Utd', 'Man Utd')
    attack_table['Squad'] = attack_table['Squad'].replace('Tottenham', 'Spurs')
    attack_table['Squad'] = attack_table['Squad'].replace('Nott\'ham Forest', 'Nott\'m Forest')
    attack_table['Squad'] = attack_table['Squad'].replace('Newcastle Utd', 'Newcastle')
    attack_table['Squad'] = attack_table['Squad'].replace('Leeds United', 'Leeds')


    final_table.drop(['name', 'position'], axis=1, inplace=True)
    def get_team_id(row):
        return id_to_team[row['opponent_team']]

    #add opponent team name column to final table
    final_table['opponent_team_name'] = final_table.apply(get_team_id, axis=1)
    final_table = pd.merge(final_table, defense_table, left_on='team', right_on='Squad', how='left')
    final_table = pd.merge(final_table, attack_table, left_on='opponent_team_name', right_on='Squad', how='left')
    final_table.drop('Squad_x', axis=1, inplace=True)
    final_table.drop('Squad_y', axis=1, inplace=True)

    #final_table = merge_attack_defense(defense_table, attack_table, final_table)
    return final_table

def merge_attack_defense(defense_table, attack_table, final_table):
    print (defense_table.Squad)
    def get_team_id(row):
        return id_to_team[row['opponent_team']]

    #add opponent team name column to final table
    final_table['opponent_team_name'] = final_table.apply(get_team_id, axis=1)
    final_table = pd.merge(final_table, defense_table, left_on='team', right_on='Squad', how='left')
    final_table = pd.merge(final_table, attack_table, left_on='opponent_team_name', right_on='Squad', how='left')
    final_table.drop('Squad_x', axis=1, inplace=True)
    final_table.drop('Squad_y', axis=1, inplace=True)

    return final_table

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

def get_fixtures():
    #Get all GWs information. And create a test data set based on the latest Gameweek.
    future_gw_csv = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/fixtures.csv"
    gws = pd.read_csv(future_gw_csv)
    #gws.dropna(inplace=True)
    gws = gws[['event', 'team_a', 'team_h']]

    def add_a_team_name(row):
        return id_to_team[row['team_a']]
    def add_h_team_name(row):
        return id_to_team[row['team_h']]

    gws['team_h_name'] = gws.apply(add_h_team_name, axis=1)
    gws['team_a_name'] = gws.apply(add_a_team_name, axis=1)

    # Convert this table into the Fbref table column names.
    gws = gws.rename(columns={'event': 'GW'})
    gws = gws.rename(columns={'team_h_name': 'team'})
    gws = gws.rename(columns={'team_a': 'opponent_team'})
    gws = gws.rename(columns={'team_a_name': 'opponent_team_name'})
    gws = gws.reindex(columns=['team', 'GW', 'opponent_team', 'opponent_team_name', 'team_h'])
    gws['was_home'] = True

    #Create another row with swapped Away and Home teams. 
    gws_swapped = pd.DataFrame({
        'team' : gws['opponent_team_name'],
        'GW' : gws['GW'],
        'opponent_team': gws['team_h'],
        'opponent_team_name': gws['team'],
        'team_h': gws['team_h'],
        'was_home': False
    })

    gws_all_fixtures = pd.concat([gws, gws_swapped], ignore_index=True)
    gws_all_fixtures.drop('team_h', axis=1, inplace=True)
    return gws_all_fixtures

'''
print("{:<20} {:<20} {:<20} {:<20}".format('CleanSheet', 'Team', 'Opposition', 'gw'))
for key, value in sorted_dict.items():
    cs, team, opp, gw = value
    print("{:<20} {:<20} {:<20} {:<20}".format(round(key,3), team, opp, gw))
predictions_info = {}
k = 0
for i,j in y_valid.iteritems():
    predictions_info[predictions_1[k]] = [j, final_table.team[i], id_to_team[X_valid_full.opponent_team[i]],X_valid_full.GW[i]]
    k = k + 1

myKeys = list(predictions_info.keys())
myKeys.sort()
sorted_dict = {i: predictions_info[i] for i in myKeys}
#print (sorted_dict)
'''