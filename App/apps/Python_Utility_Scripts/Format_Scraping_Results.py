import numpy as np
import pandas as pd
import pathlib
import os

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()
SCRAPING_PATH = PATH.joinpath("../../Data/Scraping Results").resolve()

#read scraping results
df_scraping_results = pd.read_csv(SCRAPING_PATH.joinpath('Alpine_Skiing_Scraping_Results.csv'))


##############################################################
# when the scraper returns just a year for value
# then the unbiased estimate of birthdate is the midpoint date
# ############################################################ 
def format_year_only_dates_to_midpoint(df):
    df_copy = df.copy()
    birthdates_array = np.array(df_copy['Athlete Scraped Birthdays'])
    for i in range(0,birthdates_array.shape[0]):
        if len(birthdates_array[i]) == 4:
            birthdates_array[i] = '1/7/'+birthdates_array[i]
        else:
            pass
    df_copy['Athlete Scraped Birthdays'] = birthdates_array
    return df_copy

##############################################################
# converts scraped entires into format consisitent with the 
# existing data entries
# ############################################################ 
def format_scraped_athlete_results(scraping_results):
    df_success = scraping_results.loc[(scraping_results['Athlete Scraped Ages'] != 'Unknown')|(scraping_results['Athlete Scraped Birthdays'] != 'Unknown')]
    df_success = format_year_only_dates_to_midpoint(df_success)
    df_success['Athlete Scraped Birthdays'] = pd.to_datetime(df_success['Athlete Scraped Birthdays'], infer_datetime_format=True)
    df_output_to_merge = df_success[['Athlete_ID','Athlete Scraped Birthdays']]
    return df_output_to_merge

def run():
    output = format_scraped_athlete_results(df_scraping_results)
    output.to_csv(SCRAPING_PATH.joinpath('Alpine_Skiing_Scraping_Entries_to_Merge.csv'),index=False)

if __name__ == '__main__':
    run()