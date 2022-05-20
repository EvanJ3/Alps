import pandas as pd
import numpy as np
import pathlib
import os

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()

def run():
    df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    country_json = pd.read_json(DATA_PATH.joinpath('country.json'))
    country_json = country_json[['code','name']]
    df_countries = df[['NOC','NOC Name']].groupby(by=['NOC','NOC Name'])['NOC'].count()
    df_countries = pd.DataFrame.from_records(df_countries.index, columns =['NOC', 'NOC Name'])
    merged_iso_countries = country_json.merge(df_countries,how='left',left_on='name',right_on='NOC Name')
    merged_iso_countries = merged_iso_countries.dropna()
    merged_iso_countries.to_csv(DATA_PATH.joinpath('ISO_NOC_Flag_XREF.csv'),index=False)

if __name__ == '__main__':
    run()