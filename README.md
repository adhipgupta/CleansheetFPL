# CleansheetFPL
Predict the probability of a Clean Sheet for a Premier League team for a given Fantasy Premier League Gameweek.

## Quick Start
- Train the model: 
*python3 ./trainmodel.py*
- Predict Clean Sheet for a Game week: 
*python3 predictFPLDefenderCleanSheet.py --gw {gameweek}*
  
Returns a table with the Teams and and a score of the probability of a Clean Sheet.
  
## Details
The program focuses on predicting a Clean Sheet based on the attacking and defensive style of play of the two teams. It sources the data from [Fbref](https://fbref.com/en/comps/9/Premier-League-Stats) and uses selected fields from the Goalkeeping, Advanced Goalkeeping, Shooting, Passing, Possession and other tables. 
  
The Fantasy Gameweek and Clean Sheet information is sourced from [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
  
I have created *fbrefscraper.py* that uses the Pandas framework to scrape and download the various tables in FBREF, and create the final tables with the selected columns. The Model is trained using XGBRegressor and can then be used to predict furture Gameweeks.

  

