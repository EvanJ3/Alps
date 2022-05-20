import pandas as pd
import numpy as np
import datetime as dt
import pathlib
pd.options.mode.chained_assignment = None


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()



def Tabulate_Post_Hoc_NOC_Performance(competition_ids,competition_names,return_rank=True):
    """Gets and prints the spreadsheet's header columns

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        competition_ids(list of ints): list of USOPC competition ids
        competition_names (list of str): list of USOPC competition names
        return_rank (bool): rank medal score performance in output

    Returns:
        pd.Dataframe: dataframe containing the realized olympic medal results for a country
        given the filter conditions
    """
    df_copy = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    if type(competition_ids) != list:
        competition_ids = [competition_ids] 
    df_olympics = df_copy[['Competition ID','Medal','Event Gender','Event Name Short','NOC']]
    df_olympics = df_olympics[(df_olympics['Competition ID'].isin(competition_ids))&(df_olympics['Medal']>=1)]
    df_olympics['Bronze'] = df_olympics.apply(lambda x: 1 if (x['Medal'] == 3) else 0,axis=1)
    df_olympics['Silver'] = df_olympics.apply(lambda x: 1 if (x['Medal'] == 2) else 0,axis=1)
    df_olympics['Gold'] = df_olympics.apply(lambda x: 1 if (x['Medal'] == 1) else 0,axis=1)
    
    compid_to_name = {a:b for a,b in zip(competition_ids,competition_names)}
    df_olympics['Olympics Name'] = df_olympics.apply(lambda x: compid_to_name[x['Competition ID']],axis=1)
    df_olympics_grouped = df_olympics.groupby(by=['Olympics Name','NOC']).sum()[['Bronze','Silver','Gold']].reset_index()
    if return_rank:
        df_olympics_grouped['Rank Score'] = df_olympics_grouped.apply(lambda x: x['Gold']*6.3+x['Silver']*2.4+x['Bronze']*1,axis=1)
        df_olympics_grouped['Olympic Rank'] = df_olympics_grouped.groupby(by=['Olympics Name'])['Rank Score'].rank(method='min',ascending=False).astype(int)
        df_olympics_grouped = df_olympics_grouped.sort_values(by='Olympic Rank',ascending=True)
    else:
        pass
    return df_olympics_grouped

def Generate_Olympic_Games_Options():
    """Generates dash dropdown box options for all possible Olympic
    games present in the alpine dataset treating the label as the 
    given olympic games name and the value as the competition ID

    Args:
        df(pd.Dataframe): olympic alpine dataframe

    Returns:
        list of dicts: dash dropdown formated options 
        in the following structure [{'label':x},{'value':y}]
    """
    df_copy = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    df_olympic = df_copy[['Competition City','Competition Date','Competition ID','USOC Master Competition Set Name']]
    df_olympic = df_olympic[df_olympic['USOC Master Competition Set Name'] == 'Olympic Games']
    df_olympic['Olympic Year'] = df_olympic.apply(lambda x: x['Competition Date'].split('-')[0],axis=1)
    df_olympic_grouped = df_olympic.groupby(by=['Competition City','Olympic Year','Competition ID']).count().reset_index()
    df_olympic_grouped = df_olympic_grouped.drop(columns=['USOC Master Competition Set Name','Competition Date'])
    df_olympic_grouped = df_olympic_grouped.sort_values(by='Olympic Year',ascending=False)
    df_olympic_grouped['Olympic_City_Year'] = df_olympic_grouped.apply(lambda x: x['Competition City']+' '+str(x['Olympic Year']),axis=1)
    olympic_options_list = list(df_olympic_grouped['Olympic_City_Year'])
    competition_id_list = list(df_olympic_grouped['Competition ID'])
    olympic_options = [{'label':a,'value':b} for a,b in zip(olympic_options_list,competition_id_list)]
    return olympic_options

def run():
    olympics_options = Generate_Olympic_Games_Options()
    olympic_names_list = [x['label'] for x in olympics_options]
    olympic_comp_ids_list = [x['value'] for x in olympics_options]
    olympic_results_df = Tabulate_Post_Hoc_NOC_Performance(competition_ids=olympic_comp_ids_list,competition_names=olympic_names_list,return_rank=True)
    olympic_results_df.to_csv(DATA_PATH.joinpath('Olympic_Post_Hoc_Results.csv'),index=False)
    
if __name__ == '__main__':
    run()