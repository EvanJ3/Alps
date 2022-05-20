import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import warnings
from dash import html
import pathlib

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()

def Generate_Olympic_Games_Options(df):
    """Generates dash dropdown box options for all possible Olympic
    games present in the alpine dataset treating the label as the 
    given olympic games name and the value as the competition ID

    Args:
        df(pd.Dataframe): olympic alpine dataframe

    Returns:
        list of dicts: dash dropdown formated options 
        in the following structure [{'label':x},{'value':y}]
    """
    df_copy = df.copy()
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


def Filter_Inactive_Athletes(df,base_date,inactivity_years=1):
    """Filters and returns a dataframe containing only potentially active atheltes
    by alpine discipline

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        base_data(datatime): the base date from which to measure inactivity from
        inactivity_years(int): number of years of no competitive result to consider an athlete inactive

    Returns:
        pandas.DataFrame: a reduced version of the intal dataframe including
        only active athletes
    """
    df_copy = df.copy()
    athlete_df = df_copy[['Competition Date','Event Gender','Event Name Short','Person ID','Competitor','Athlete Birth Date','NOC']]
    athlete_df['Competition Date'] = pd.to_datetime(athlete_df['Competition Date'].copy())
    athlete_df['Athlete Birth Date'] = pd.to_datetime(athlete_df['Athlete Birth Date'].copy())
    grouped_athlete_df = athlete_df.groupby(by=['Person ID','Athlete Birth Date','Event Gender','Event Name Short','NOC']).max(['Competition Date']).reset_index()
    split_base_date = base_date.split('-')
    inactivity_days = inactivity_years*365.25
    base_datetime = dt.datetime(int(split_base_date[0]),int(split_base_date[1]),int(split_base_date[2]))
    threshold_date = str(pd.to_datetime(base_datetime - dt.timedelta(days=inactivity_days)))
    grouped_athlete_df = grouped_athlete_df[(grouped_athlete_df['Competition Date']>=threshold_date)&(grouped_athlete_df['Competition Date']<base_datetime)]
    if grouped_athlete_df.empty:
        grouped_athlete_df = None
    else:
        grouped_athlete_df['Age_Years_At_Olympic_Start'] = grouped_athlete_df.apply(lambda x: (base_datetime-x['Athlete Birth Date']).days/365.25,axis=1)
        grouped_athlete_df = grouped_athlete_df[(grouped_athlete_df['Age_Years_At_Olympic_Start']>15.0)&(grouped_athlete_df['Age_Years_At_Olympic_Start']<=45.0)]
        grouped_athlete_df = grouped_athlete_df.drop(columns=['Athlete Birth Date','Competition Date'])
    return grouped_athlete_df


def Get_Olympics_Potential_Active_Athletes(df,competition_id,competition_name):
    """Top level function which returns potentially active athletes for a 
    given compeition_id and name; composed of the previous filter function

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        competition_id(int): USOPC competition id
        competition_name(str): UOSPC competition name

    Returns:
        pandas.Dataframe: returns a filtered dataframe containing only potentailly active
        athletes for a given competition
    """
    df_copy = df.copy()
    olympic_event_list = ['Downhill','Super G','Giant Slalom','Slalom','Combination']
    df_olympics = df_copy[['Competition ID','Competition Date','NOC']]
    df_olympics = df_olympics[df_olympics['Competition ID']==competition_id]
    olympic_start_date = df_olympics.sort_values(by='Competition Date')['Competition Date'].iloc()[0]
    pre_olympic_events = df_copy[(df_copy['Competition Date']<olympic_start_date)&(df_copy['Event Name Short'].isin(olympic_event_list))&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')]
    if int(competition_name.split(' ')[-1])<1966:
        inactivity_years = 4
    else:
        inactivity_years = 1.5
    potential_athletes_df = Filter_Inactive_Athletes(df=pre_olympic_events,base_date=olympic_start_date,inactivity_years=inactivity_years)
    if potential_athletes_df is not None:
        potential_athletes_df['Olympics Name'] = potential_athletes_df.apply(lambda x: competition_name,axis=1)
    return potential_athletes_df

def lookup_peak_age(age_dict,gender,event_name,mode='upper'):
    """Gets and returns a given modeled peak age from modeling results

    Args:
        age_dict (dict): age dictrionary as generated by Get_Event_Gender_Peak_Age_Bounds()
        gender(str): event gender ['Men' Women']
        event_name (str): event name short/ alpine discipline
        mode(str) = lookup age lower bound or upper options ["lower","upper"]

    Returns:
        float: peak age range selected
    """
    level_1 = age_dict[gender]
    level_2 = level_1[event_name]
    if mode == 'lower':
        level_3 = level_2['Age Lower']
    else:
        level_3 = level_2['Age Upper']
    return level_3

def Tabulate_Country_Age_Counts(df,competition_ids,competition_names,age_dict):
    """Gets and prints the spreadsheet's header columns

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        competition_ids (list of ints): USOPC competition ids
        competition_names (list of strings): USOPC competition names
        age_dict (dict): age dictrionary as generated by Get_Event_Gender_Peak_Age_Bounds()

    Returns:
        pd.Dataframe: dataframe containing filtered event and competition level age counts 
        pd.Dateframe: dataframe containing the corresponding athlete profiles
    """
    df_list = []
    df_copy = df.copy()
    for i in range(0,len(competition_ids)):
        df_olympic_i = Get_Olympics_Potential_Active_Athletes(df=df_copy,competition_id=competition_ids[i],competition_name=competition_names[i])
        if df_olympic_i is not None:
            df_list.append(df_olympic_i)
        else:
            pass
    if len(df_list) >1:
        df_concat = pd.concat(df_list)
    else:
        df_concat = df_list[i]
    df_concat['Pre-Peak Age Athletes'] = df_concat.apply(lambda x: 1 if x['Age_Years_At_Olympic_Start'] < lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'lower') else 0,axis=1)
    df_concat['Peak Age Athletes'] = df_concat.apply(lambda x: 1 if (x['Age_Years_At_Olympic_Start'] >= lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'lower') ) and (x['Age_Years_At_Olympic_Start'] <= lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'upper') ) else 0,axis=1)
    df_concat['Post-Peak Age Athletes'] = df_concat.apply(lambda x: 1 if x['Age_Years_At_Olympic_Start'] > lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'upper') else 0,axis=1)
    df_athletes = df_concat.copy()
    df_grouped = df_concat.groupby(by=['Olympics Name','NOC','Event Gender','Event Name Short']).sum()[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']].reset_index()
    df_grouped['Total NOC Event Gender Potential_Athletes'] = df_grouped.apply(lambda x: x['Pre-Peak Age Athletes']+x['Peak Age Athletes']+x['Post-Peak Age Athletes'],axis=1)
    df_grouped['Event Gender Pipeline Strength'] = df_grouped.apply(lambda x: 0.25*x['Pre-Peak Age Athletes']+0.5*x['Peak Age Athletes']+0.25*x['Post-Peak Age Athletes'],axis=1)
    df_olympics_gender_event_grouped = df_grouped.groupby(by=['Olympics Name','Event Gender','Event Name Short'])['Event Gender Pipeline Strength']
    event_gender_min,event_gender_max = df_olympics_gender_event_grouped.transform('min'),df_olympics_gender_event_grouped.transform('max')
    df_grouped['Scaled Event Gender Pipeline Strength'] = (df_grouped['Event Gender Pipeline Strength'] - event_gender_min) / (event_gender_max - event_gender_min)
    df_grouped['Scaled Event Gender Pipeline Strength'] =df_grouped['Scaled Event Gender Pipeline Strength'].fillna(0)
    df_noc_olympics_grouped = df_grouped.groupby(by=['Olympics Name','NOC'])['Scaled Event Gender Pipeline Strength']
    df_grouped['Overall Pipeline Strength'] = df_noc_olympics_grouped.transform('sum')
    df_olympics_grouped = df_grouped.groupby(by=['Olympics Name'])['Overall Pipeline Strength']
    overall_rank_min,overall_rank_max = df_olympics_grouped.transform('min'),df_olympics_grouped.transform('max')
    df_grouped['Scaled Olympic Pipeline Strength'] = (df_grouped['Overall Pipeline Strength'] - overall_rank_min) / (overall_rank_max - overall_rank_min)
    df_grouped['Scaled Olympic Pipeline Strength'] = df_grouped['Scaled Olympic Pipeline Strength'].fillna(0)
    df_grouped['Event Gender Pipeline Rank'] = df_grouped.groupby(by=['Olympics Name','Event Gender','Event Name Short'])['Scaled Event Gender Pipeline Strength'].rank(method='min',ascending=False).astype(int)
    df_reduced = df_grouped[['Olympics Name','NOC','Scaled Olympic Pipeline Strength']].drop_duplicates()
    df_reduced['Overall Pipeline Rank'] = df_reduced.groupby(by=['Olympics Name'])['Scaled Olympic Pipeline Strength'].rank(method='min',ascending=False).astype(int)
    df_grouped= df_grouped.merge(df_reduced, left_on=['Olympics Name','NOC'],right_on=['Olympics Name','NOC'],how='left')
    df_grouped['Olympic Year'] = df_grouped.apply(lambda x: int(x['Olympics Name'].split(' ')[-1]),axis=1)
    df_grouped = df_grouped.sort_values(by=['Olympic Year','Overall Pipeline Rank','Event Gender Pipeline Rank'],ascending=[False,True,True])
    return df_grouped,df_athletes

def Get_Event_Gender_Peak_Age_Bounds():
    """Generates a dictionary of the peak age bounds based on the 
    event kde analysis perspective

    Args:
        function takes no arguments

    Returns:
        dict: returns dict containing event and gender paired peak age ranges
    """
    df = pd.read_csv(DATA_PATH.joinpath('KDE_RESULT_DF.csv'))
    age_dict = dict()
    unique_genders = list(df['Genders'].unique())
    unique_events = list(df['Events'].unique())
    for i in range(0,len(unique_genders)):
        gender_i = unique_genders[i]
        gender_dict_i = dict()
        for j in range(0,len(unique_events)):
            event_j = unique_events[j]
            events_j_dict = dict()
            filterd_df_ij = df[(df['Events']==event_j)&(df['Genders']==gender_i)]
            events_j_dict['Age Lower'] = filterd_df_ij['Top Age Lower'].iloc()[0]
            events_j_dict['Age Upper'] = filterd_df_ij['Top Age Upper'].iloc()[0]
            gender_dict_i[event_j] = events_j_dict
        age_dict[gender_i] = gender_dict_i
    return age_dict

def Generate_Olympic_Event_Data(df,competition_ids,competition_names):
    """aggregates and generates the olympic and athlete frames for 
    historic olympic competitions

    Args:
        df(dataframe): alpine event dataframe
        competition_ids (list of ints): USOPC competition ids
        competition_names (list of strings): USOPC competition names

    Returns:
        pd.Dataframe: dataframe containing filtered event and competition level age counts 
        pd.Dateframe: dataframe containing the corresponding athlete profiles
    """
    Event_Gender_Age_Bounds_Dict = Get_Event_Gender_Peak_Age_Bounds()
    olympic_df,athlete_df = Tabulate_Country_Age_Counts(df,competition_ids,competition_names,age_dict=Event_Gender_Age_Bounds_Dict)
    return olympic_df,athlete_df

def Generate_Future_Olympic_Event_Data():
    """aggregates and generates the olympic and athlete frames for 
    future olympic competitions

    Args:
        function takes no arguments

    Returns:
        pd.Dataframe: dataframe containing filtered event and competition level age counts 
        pd.Dateframe: dataframe containing the corresponding athlete profiles
    """
    df_list = []
    df_copy = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    competition_names = ['Milan Cortina 2026']
    competition_dates = ['2026-02-06']
    olympic_event_list = ['Downhill','Super G','Giant Slalom','Slalom','Combination']
    current_date = dt.datetime.now()
    age_dict = Get_Event_Gender_Peak_Age_Bounds()
    for i in range(0,len(competition_names)):
        olympic_start_date = competition_dates[i]
        pre_olympic_events = df_copy[(df_copy['Event Name Short'].isin(olympic_event_list))&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')]
        split_oly_date = olympic_start_date.split('-')
        oly_start_year = int(split_oly_date[0])
        inactivity_years = oly_start_year - int(dt.date.today().strftime("%Y")) + 2
        df_olympic_i = Filter_Inactive_Athletes(df=pre_olympic_events,base_date=olympic_start_date,inactivity_years=inactivity_years)
        if df_olympic_i is not None:
            df_olympic_i['Olympics Name'] = df_olympic_i.apply(lambda x: competition_names[i],axis=1)
        if df_olympic_i is not None:
            df_list.append(df_olympic_i)
        else:
            pass
    if len(df_list) >1:
        df_concat = pd.concat(df_list)
    else:
        df_concat = df_list[i]
    
    df_concat['Pre-Peak Age Athletes'] = df_concat.apply(lambda x: 1 if x['Age_Years_At_Olympic_Start'] < lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'lower') else 0,axis=1)
    df_concat['Peak Age Athletes'] = df_concat.apply(lambda x: 1 if (x['Age_Years_At_Olympic_Start'] >= lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'lower') ) and (x['Age_Years_At_Olympic_Start'] <= lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'upper') ) else 0,axis=1)
    df_concat['Post-Peak Age Athletes'] = df_concat.apply(lambda x: 1 if x['Age_Years_At_Olympic_Start'] > lookup_peak_age(age_dict,x['Event Gender'],x['Event Name Short'],'upper') else 0,axis=1)
    df_athletes = df_concat.copy()
    df_grouped = df_concat.groupby(by=['Olympics Name','NOC','Event Gender','Event Name Short']).sum()[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']].reset_index()
    df_grouped['Total NOC Event Gender Potential_Athletes'] = df_grouped.apply(lambda x: x['Pre-Peak Age Athletes']+x['Peak Age Athletes']+x['Post-Peak Age Athletes'],axis=1)
    df_grouped['Event Gender Pipeline Strength'] = df_grouped.apply(lambda x: 0.25*x['Pre-Peak Age Athletes']+0.5*x['Peak Age Athletes']+0.25*x['Post-Peak Age Athletes'],axis=1)
    df_olympics_gender_event_grouped = df_grouped.groupby(by=['Olympics Name','Event Gender','Event Name Short'])['Event Gender Pipeline Strength']
    event_gender_min,event_gender_max = df_olympics_gender_event_grouped.transform('min'),df_olympics_gender_event_grouped.transform('max')
    df_grouped['Scaled Event Gender Pipeline Strength'] = (df_grouped['Event Gender Pipeline Strength'] - event_gender_min) / (event_gender_max - event_gender_min)
    df_grouped['Scaled Event Gender Pipeline Strength'] =df_grouped['Scaled Event Gender Pipeline Strength'].fillna(0)
    df_noc_olympics_grouped = df_grouped.groupby(by=['Olympics Name','NOC'])['Scaled Event Gender Pipeline Strength']
    df_grouped['Overall Pipeline Strength'] = df_noc_olympics_grouped.transform('sum')
    df_olympics_grouped = df_grouped.groupby(by=['Olympics Name'])['Overall Pipeline Strength']
    overall_rank_min,overall_rank_max = df_olympics_grouped.transform('min'),df_olympics_grouped.transform('max')
    df_grouped['Scaled Olympic Pipeline Strength'] = (df_grouped['Overall Pipeline Strength'] - overall_rank_min) / (overall_rank_max - overall_rank_min)
    df_grouped['Scaled Olympic Pipeline Strength'] = df_grouped['Scaled Olympic Pipeline Strength'].fillna(0)
    df_grouped['Event Gender Pipeline Rank'] = df_grouped.groupby(by=['Olympics Name','Event Gender','Event Name Short'])['Scaled Event Gender Pipeline Strength'].rank(method='min',ascending=False).astype(int)
    df_reduced = df_grouped[['Olympics Name','NOC','Scaled Olympic Pipeline Strength']].drop_duplicates()
    df_reduced['Overall Pipeline Rank'] = df_reduced.groupby(by=['Olympics Name'])['Scaled Olympic Pipeline Strength'].rank(method='min',ascending=False).astype(int)
    df_grouped= df_grouped.merge(df_reduced, left_on=['Olympics Name','NOC'],right_on=['Olympics Name','NOC'],how='left')
    df_grouped['Olympic Year'] = df_grouped.apply(lambda x: int(x['Olympics Name'].split(' ')[-1]),axis=1)
    df_grouped = df_grouped.sort_values(by=['Olympic Year','Overall Pipeline Rank','Event Gender Pipeline Rank'],ascending=[False,True,True])
    return df_grouped,df_athletes


def run():
    df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    df = df[df['Class'] == 'Elite']
    olympics_options = Generate_Olympic_Games_Options(df)
    olympic_names_list = [x['label'] for x in olympics_options]
    olympic_comp_ids_list = [x['value'] for x in olympics_options]
    comp_ids_to_olympic_names = {a:b for a,b in zip(olympic_comp_ids_list,olympic_names_list)}
    olympic_names_to_comp_ids = {b:a for a,b in zip(olympic_comp_ids_list,olympic_names_list)}
    olympic_ranking_data,olympic_athlete_data = Generate_Olympic_Event_Data(df,olympic_comp_ids_list,olympic_names_list)
    future_olympic_ranking_data,future_olympic_athlete_data = Generate_Future_Olympic_Event_Data()
    olympic_data_ranking_concat = pd.concat([future_olympic_ranking_data,olympic_ranking_data])
    olympic_athlete_data_concat = pd.concat([future_olympic_athlete_data,olympic_athlete_data])
    olympic_data_ranking_concat.to_csv(DATA_PATH.joinpath('Olympic_Ranking_Data.csv'),index=False)
    olympic_athlete_data_concat.to_csv(DATA_PATH.joinpath('Olympic_Athlete_Data.csv'),index=False)


if __name__ == '__main__':
    run()