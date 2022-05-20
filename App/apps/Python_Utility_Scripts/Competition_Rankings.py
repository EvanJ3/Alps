import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import datetime as dt
import math
import pathlib
import os

pd.options.mode.chained_assignment = None

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()

def Get_Num_Competition_Participants(event_hash,df):
    """Calculates the total number of event participants
    in a given compeition

    Args:
        event hash (str): unique event identifier
        df (dataframe): the olympic event dataframe

    Returns:
        int: number of competition participants
    """
    df_reduced = df[df['Event Hash'] == event_hash]
    return df_reduced.shape[0]

def Get_Num_Non_DNF_Competitors(event_hash,df):
    """Calculates the number of event competitiors
    in a given event without including those athletes that
    recieved a DNF or similar non finishing positon

    Args:
        event hash (str): unique event identifier
        df (dataframe): the olympic event dataframe

    Returns:
        int: number of non-df competition participants
    """
    df_reduced = df[df['Event Hash'] == event_hash]
    num_all_competitors = df_reduced.shape[0]
    df_dnf = df_reduced[df_reduced['Rank']==-1]
    num_dnf_competitors = df_dnf.shape[0]
    num_non_dnf_competitors = num_all_competitors - num_dnf_competitors
    return num_non_dnf_competitors

def Get_Season_Sport_Rankings(df,event_name_short,event_gender,event_season,top_n=20):
    """Locates and returns the top_n highest ranked athletes for a given
        event, season, and gender and returns their ids and ranks

    Args:
        df (dataframe): the olympic event dataframe
        event_name_short(string): the alpine event shortname
        event_gender(string): the event or athlete gender
        event_season(int): the season of interest
        top_n(int): number of top athletes to include in output

    Returns:
        list: athlete ids
        list: athlete competitive rankings
    """
    reduced_rankings_df = df[(df['Event Name Short']==event_name_short)&(df['Event Gender']==event_gender)&(df['Season'] == event_season)&(df['USOC Master Competition Set Name']=='Standing/Ranking List')&(df['Rank']<=top_n)]
    reduced_rankings_df =reduced_rankings_df[['Person ID','Rank']]
    reduced_rankings_df = reduced_rankings_df.sort_values(by='Rank')
    top_n_athlete_ids = list(reduced_rankings_df['Person ID'])
    top_n_athlete_ranks = list(reduced_rankings_df['Rank'])
    return top_n_athlete_ids, top_n_athlete_ranks


def Lookup_Event_Competitors(df,event_hash):
    """Looks up the competitor names of athletes in a given event

    Args:
        event hash (str): unique event identifier
        df (dataframe): the olympic event dataframe

    Returns:
        list: athlete ids
    """
    reduced_event_df = df[(df['Event Hash']==event_hash)]
    event_athlete_ids = list(reduced_event_df['Person ID'].unique())
    return event_athlete_ids


def Lookup_Event_Hash_Details(df,event_hash):
    """Lookups up a unique event and returns its coresponding
    identifying features

    Args:
        event hash (str): unique event identifier
        df (dataframe): the olympic event dataframe

    Returns:
        str: Event's gender class
        int: Event's season of occurance
        str: Event name
        str: Event class level
    """
    reduced_event_df = df[(df['Event Hash']==event_hash)]
    reduced_event_df = reduced_event_df[['Event Gender','Event Name Short','Season','Class']]
    gender = reduced_event_df['Event Gender'].iloc()[0]
    season = reduced_event_df['Season'].iloc()[0]
    event_name = reduced_event_df['Event Name Short'].iloc()[0]
    class_level = reduced_event_df['Class'].iloc()[0]
    return gender,season,event_name,class_level

    
def Threshold_WR_Ranking(value):
    """Applies weight to a given world ranking for calculation of
    comeptition difficulty

    Args:
        value(int): world ranking

    Returns:
        float: weighted world ranking value
    """
    if value <= 5:
        output = 1.0
    elif (value > 5) & (value<=10):
        output = 0.5
    elif (value >10) & (value<=20):
        output = 0.1
    else:
        output = 0.0
    return output

def Score_Compition_WR_Difficulty(df,event_hash,top_n=20):
    """Creates difficulty score for world rankinig

    Args:
        event hash (str): unique event identifier
        df (dataframe): the olympic event dataframe
        top_n (int): number of wr positions to consider

    Returns:
        float: difficulty of compeitition based on world ranked compeititors
    """
    gender,season,event,class_level = Lookup_Event_Hash_Details(df,event_hash)
    if class_level == 'Elite':
        top_n_athlete_ids, top_n_athlete_ranks = Get_Season_Sport_Rankings(df,event_name_short=event,event_gender=gender,event_season=season,top_n=top_n)
        event_athlete_ids = Lookup_Event_Competitors(df,event_hash)
        difficulty = 0.0
        for i in range(0,len(top_n_athlete_ids)):
            if top_n_athlete_ids[i] in event_athlete_ids:
                difficulty+= Threshold_WR_Ranking(top_n_athlete_ranks[i])
            else:
                pass
    else:
        difficulty = 0.0
        
    return difficulty

def event_time_string_to_seconds(value):
    """formats event time enties into consistent time format of seconds

    Args:
        value: finish time string

    Returns:
        float: time in seconds
    """
    minutes = value.split(':')[0]
    if minutes == value:
        minutes = 0
        seconds = int(value.split('.')[0])
    else:
        minutes = int(minutes)*60
        first_parse = value.split('.')[0]
        seconds = int(first_parse.split(':')[-1])
        
    tenth_seconds = int(value.split('.')[-1])/10
    output = seconds+minutes+tenth_seconds
    
    return output

def min_max_scaling(x,x_min,x_max):
    """min-max scales an input value based on provided minimum and maximum value

    Args:
        x(float or int): value to scale
        x_max (float or int): maximum value in scale range
        x_min (float or int):minimum value in scale range

    Returns:
        float: min-max scaled x value
    """
    return (x-x_min)/(x_max-x_min)

def scale_comp_participant_difficulty(lookup_df,event,gender,class_level,value):
    """Min-max scales competition participant difficulty

    Args:
         lookup_df(dataframe):
         event(str): USOPC event name short
         gender(str): UOSPC gender
         class_level(str): alpine class to operate over (elite, junior, youth)
         value(float/int): compeitive value to scale

    Returns:
        float: returns scaled particaptn compeittion value
    """
    reduced_lookup_df = lookup_df[(lookup_df['Event Name Short']==event)&(lookup_df['Event Gender']==gender)&(lookup_df['Class']==class_level)]
    max_val =reduced_lookup_df['max_participants'].iloc()[0]
    min_val =reduced_lookup_df['min_participants'].iloc()[0]
    scaled_val = min_max_scaling(value,min_val,max_val)
    return scaled_val

def scale_comp_wr_difficulty(lookup_df,event,gender,class_level,value):
    """Min-max scales world ranking competition difficulty

    Args:
         lookup_df(dataframe):
         event(str): USOPC event name short
         gender(str): UOSPC gender
         class_level(str): alpine class to operate over (elite, junior, youth)
         value(float/int): compeitive value to scale

    Returns:
        float: returns scaled world ranking difficulty value
    """
    reduced_lookup_df = lookup_df[(lookup_df['Event Name Short']==event)&(lookup_df['Event Gender']==gender)&(lookup_df['Class']==class_level)]
    max_val =reduced_lookup_df['max_wr'].iloc()[0]
    min_val =reduced_lookup_df['min_wr'].iloc()[0]
    scaled_val = min_max_scaling(value,min_val,max_val)
    return scaled_val

def scale_comp_std_difficulty(lookup_df,event,gender,class_level,value):
    """Min-max scales competition standard deveiation of finish time difficulty

    Args:
         lookup_df(dataframe):
         event(str): USOPC event name short
         gender(str): UOSPC gender
         class_level(str): alpine class to operate over (elite, junior, youth)
         value(float/int): compeitive value to scale

    Returns:
        float: returns scaled compeition standard deveiation of finsh time diffculty value
    """
    reduced_lookup_df = lookup_df[(lookup_df['Event Name Short']==event)&(lookup_df['Event Gender']==gender)&(lookup_df['Class']==class_level)]
    max_val =reduced_lookup_df['max_std'].iloc()[0]
    min_val =reduced_lookup_df['min_std'].iloc()[0]
    scaled_val = min_max_scaling(value,min_val,max_val)
    scaled_val = abs(scaled_val -1)
    return scaled_val

def scale_comp_difficulty(lookup_df,event,gender,class_level,value):
    """Min-max scales competition difficulty

    Args:
         lookup_df(dataframe):
         event(str): USOPC event name short
         gender(str): UOSPC gender
         class_level(str): alpine class to operate over (elite, junior, youth)
         value(float/int): compeitive value to scale

    Returns:
        float: returns scaled compeittion difficulty value
    """
    reduced_lookup_df = lookup_df[(lookup_df['Event Name Short']==event)&(lookup_df['Event Gender']==gender)&(lookup_df['Class']==class_level)]
    max_val =reduced_lookup_df['max_difficulty'].iloc()[0]
    min_val =reduced_lookup_df['min_difficulty'].iloc()[0]
    scaled_val = min_max_scaling(value,min_val,max_val)
    return scaled_val

def Generate_Competition_Difficulty_Table():
    """Creates and writes the Timed_Difficulty.csv table via 
    transformations and operations on the base cleaned alpine dataset

    Args:
       no input arguments

    Returns:
        returns no variables or objects
    """
    df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    comp_df = df[['Event Hash','Competition Date','Competition ID','Competition City','Event Name Short','Event Gender','Class','Season','Result Type']]
    comp_df = comp_df.drop_duplicates()
    comp_df['Num Competitors'] = df.apply(lambda x: Get_Num_Competition_Participants(x['Event Hash'],df),axis=1)
    comp_df['Num Non DNF Competitors'] = df.apply(lambda x: Get_Num_Non_DNF_Competitors(x['Event Hash'],df),axis=1)
    comp_df = comp_df[(comp_df['Num Competitors']>5)&(comp_df['Class']=='Elite')]
    timed_comp_df = comp_df[comp_df['Result Type'] == 'Time']
    points_comp_df = comp_df[comp_df['Result Type'] == 'Points']
    timed_comp_df['Comp_WR_Rating'] = timed_comp_df.apply(lambda x: Score_Compition_WR_Difficulty(df,event_hash=x['Event Hash'],top_n=20),axis=1)
    df_times = df[(df['Result Type'] == 'Time')&(df['Result'] != '999.99')&(df['Result'] != 'Unknown')]
    df_times = df_times[['Event Hash','Result']]
    df_times['Result Seconds'] = df_times.apply(lambda x: event_time_string_to_seconds(x['Result']),axis=1)
    top_percent = .4
    unique_event_hashes = np.array(df_times['Event Hash'].unique())
    std_list = []
    for i in range(0,unique_event_hashes.shape[0]):
        df_copy = df_times.copy()
        df_copy = df_copy[df_copy['Event Hash']==unique_event_hashes[i]]
        df_copy = df_copy.sort_values('Result Seconds')
        num_entries = df_copy.shape[0]
        top_n = math.ceil(num_entries*top_percent)
        df_copy = df_copy.iloc()[0:top_n]
        top_percent_avg_time = df_copy['Result Seconds'].mean()
        top_percent_std_time = df_copy['Result Seconds'].std()
        std_list.append(top_percent_std_time)

    std_array = np.array(std_list)
    std_df = pd.DataFrame(std_array,unique_event_hashes)
    std_df = std_df.reset_index()
    std_df.rename(columns={0:'Event Std Seconds','index':'Event Hash'},inplace=True)
    timed_comp_df = timed_comp_df.merge(std_df,how='left',on='Event Hash')
    timed_comp_grouped = timed_comp_df.groupby(by=['Event Name Short','Event Gender','Class']).agg(max_participants=('Num Competitors',max),min_participants=('Num Competitors',min),max_wr=('Comp_WR_Rating',max),min_wr=('Comp_WR_Rating',min),min_std=('Event Std Seconds',min),max_std=('Event Std Seconds',max))
    timed_comp_grouped = timed_comp_grouped.reset_index()
    timed_comp_grouped['min_std'].fillna(0.0,inplace=True)
    timed_comp_grouped['max_std'].fillna(0.0,inplace=True)
    timed_comp_df['Scaled Num Participants'] = timed_comp_df.apply(lambda x:scale_comp_participant_difficulty(lookup_df=timed_comp_grouped,event=x['Event Name Short'],gender=x['Event Gender'],class_level=x['Class'],value=x['Num Competitors']),axis=1)
    timed_comp_df['Scaled WR Difficulty'] = timed_comp_df.apply(lambda x:scale_comp_wr_difficulty(lookup_df=timed_comp_grouped,event=x['Event Name Short'],gender=x['Event Gender'],class_level=x['Class'],value=x['Comp_WR_Rating']),axis=1)
    timed_comp_df['Scaled STD Difficulty'] = timed_comp_df.apply(lambda x:scale_comp_std_difficulty(lookup_df=timed_comp_grouped,event=x['Event Name Short'],gender=x['Event Gender'],class_level=x['Class'],value=x['Event Std Seconds']),axis=1)
    timed_comp_df['Scaled Num Participants'].fillna(0.0,inplace=True)
    timed_comp_df['Scaled WR Difficulty'].fillna(0.0,inplace=True)
    timed_comp_df['Scaled STD Difficulty'].fillna(0.0,inplace=True)
    timed_comp_df['Difficulty Rating'] = timed_comp_df.apply(lambda x:(0.25*x['Scaled Num Participants'])+(x['Scaled WR Difficulty']*0.5)+(x['Scaled STD Difficulty']*0.25),axis=1)
    timed_comp_df.sort_values('Difficulty Rating',ascending=False)
    grouped_difficulty_rating_df = timed_comp_df.groupby(by=['Event Name Short','Event Gender','Class']).agg(max_difficulty=('Difficulty Rating',max),min_difficulty=('Difficulty Rating',min))
    grouped_difficulty_rating_df = grouped_difficulty_rating_df.reset_index()
    timed_comp_df['Scaled Difficulty'] = timed_comp_df.apply(lambda x:scale_comp_difficulty(lookup_df=grouped_difficulty_rating_df,event=x['Event Name Short'],gender=x['Event Gender'],class_level=x['Class'],value=x['Difficulty Rating']),axis=1)
    timed_comp_df[(timed_comp_df['Event Name Short'] == 'Slalom')&(timed_comp_df['Event Gender']=='Women')].sort_values('Scaled Difficulty',ascending=False)
    timed_comp_df= timed_comp_df[['Event Hash','Num Competitors','Num Non DNF Competitors','Comp_WR_Rating','Event Std Seconds','Scaled Num Participants','Scaled WR Difficulty','Scaled STD Difficulty','Difficulty Rating','Scaled Difficulty']]
    timed_comp_df.to_csv(DATA_PATH.joinpath('Timed_Competition_Difficulty.csv'),index=False)

if __name__ == '__main__':
    Generate_Competition_Difficulty_Table()