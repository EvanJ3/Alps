import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import pathlib


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()
SCRAPE_DATA_PATH = PATH.joinpath("../../Data/Scraping Results").resolve()


########################################################
# pandas lambda function for string replacement
########################################################
def convert_empty_strings(input_string,replacement_string='Unknown'):
    if input_string == '':
        return replacement_string
    else:
        return input_string

########################################################
# standarizes medal variagle to integer encoding
########################################################
def convert_medal_to_integer(medal_string):
    medal_dict = {
        '':0,
        'B':3,
        'S':2,
        'G':1
    }
    return medal_dict[medal_string]

########################################################
# cleans missing result entires 
########################################################
def clean_missing_result_entries(df):
    df_copy = df.copy()
    result_status_entries = np.array(df_copy['Result Status'])
    result_entries = np.array(df_copy['Result'])
    result_type = np.array(df_copy['Result Type'])
    for i in range(0,result_status_entries.shape[0]):
        if (result_status_entries[i] == '') and (result_entries[i] == '') and (result_type[i] == ''):
            result_status_entries[i] = 'Unknown'
            result_entries[i] = 'Unknown'
            result_type[i] = 'Unknown'
        elif (result_status_entries[i] == '') and (result_entries[i] == ''):
            result_status_entries[i] = 'Unknown'
            result_entries[i] = 'Unknown'
        elif (result_type[i] == '') and (result_entries[i] == ''):
            result_entries[i] = 'Unknown'
            result_type[i] = 'Unknown'
        elif (result_entries[i] == '') and (result_type[i] == 'Points'):
            result_entries[i] = '0.0'
        elif (result_entries[i] == '') and (result_type[i] == 'Time'):
            result_entries[i] = '999.99'
        else:
            pass
            
    df_copy['Result Status'] = result_status_entries
    df_copy['Result'] = result_entries
    df_copy['Result Type'] = result_type
    return df_copy

########################################################
# standarizes treatment and convention of season notation
########################################################
def combine_split_seasons(df):
    df_copy = df.copy()
    season_array = np.array(df_copy['Season'])
    for i in range(season_array.shape[0]):
        if '/' in season_array[i]:
            season_array[i] = season_array[i].split('/')[0]
        else:
            pass
    df_copy['Season'] = season_array
    return df_copy

########################################################
# replace exact entry value
########################################################
def exact_replace_entry(x,value,replacement_value):
    if x==value:
        return replacement_value
    else:
        return x

########################################################
# standardized treatment of multiple dnf types
########################################################
def rank_dnf_formater(result_status,rank):
    if (rank == '') & (result_status in ['DNF','DNS','DSQ']):
        output = -1
    else:
        output = rank
    return output


########################################################
# unify restlt status encoding
########################################################
def format_missing_result_status(result_status,rank):
    if (result_status == 'Unknown') & (rank not in [-1,'','-1']):
        output = 'FIN'
    else:
        output = result_status
    return output


########################################################
# standardize world ranking city entires which previously were none
########################################################       
def format_world_ranking_city_entires(usoc_comp_set_type,city):
    if (usoc_comp_set_type == 'Standing/Ranking List'):
        output = "N/A"
    else:
        output = city
    return output


########################################################
# standardizes treatment of missing comp dates
########################################################
def correct_missing_comp_dates(date,season):
    if (date== 'Unknown')&(season!=''):
        output = pd.to_datetime('12/01/'+str(season), infer_datetime_format=True)
    else:
        output = date
    return output


########################################################
# finds athelte aliases
########################################################
def Return_Alias_Names_From_Person_ID(df,person_id):
    df_copy = df[df['Person ID']==person_id]
    df_copy = df_copy[df_copy['Competition Date']!= 'Unknown']
    most_recent_name = df_copy.sort_values('Competition Date',ascending=False)['Competitor'].iloc()[0]
    names = np.unique(df_copy['Competitor'])
    aliases = names[names!=most_recent_name]
    
    if names.shape[0] > 1:
        output = ';'.join(aliases)
    else:
        output = ''
    return output


########################################################
# dictionary of athlete aliases containing only
# athletes with multiple names, but singluar current name
########################################################
def Return_Most_Recent_Athlete_Name_Dict(df):
    df_copy = df[['Person ID','Competitor','Competition Date']]
    df_copy = df_copy[df_copy['Competition Date']!= 'Unknown']
    unique_ids = np.array(df_copy['Person ID'].unique())
    person_id_dict = dict()
    for i in range(0,unique_ids.shape[0]):
        person_id_i = unique_ids[i]
        df_copy2 = df_copy[df_copy['Person ID'] == person_id_i]
        most_recent_name = df_copy2.sort_values('Competition Date',ascending=False)['Competitor'].iloc()[0]
        person_id_dict[person_id_i] = most_recent_name
    return person_id_dict


########################################################
# converts das in age to a birthdate
########################################################
def age_days_to_birthday(event_date,age_days):
    if age_days != 'Unknown':
        age_days = int(float(age_days))
        time_diff = datetime.timedelta(days=age_days)
        output = str(event_date-time_diff)
    else:
        output = 'Unknown'
    return output


########################################################
# reconciles multiple athlete derived birthdates
# takes mode of date realized
########################################################
def reconcile_multiple_athlete_birthdays(solo_df):
    df = solo_df.copy()
    athlete_prior_birth_dates = np.array(df['Athlete Birth Date'].values)
    athlete_new_birth_dates = athlete_prior_birth_dates.copy()
    athlete_ids = np.array(df['Person ID'].values)
    unique_athlete_ids = np.unique(athlete_ids)
    for i in range(0,unique_athlete_ids.shape[0]):
        athlete_i_entries_df = df[df['Person ID'] == unique_athlete_ids[i]]
        date_value_counts_object_i = athlete_i_entries_df['Athlete Birth Date'].value_counts()
        mode_i = date_value_counts_object_i.index[0]
        if (mode_i == 'Unknown') and (date_value_counts_object_i.index.shape[0]>1):
            mode_i = date_value_counts_object_i.index[1]
        else:
            pass
        athlete_i_idxs = np.argwhere(athlete_ids==unique_athlete_ids[i])
        athlete_new_birth_dates[athlete_i_idxs] = mode_i
    df['Athlete Birth Date'] = athlete_new_birth_dates
    return df


########################################################
# identify and group missing athlete entires (age)
########################################################
def find_missing_athletes(solo_events_df,write_output=False):
    missing_bdays = solo_events_df[solo_events_df['Athlete Birth Date'] == 'Unknown']
    missing_person_bdays_idx = missing_bdays.groupby(by=['Person ID','Competitor'])['Athlete Birth Date'].count().index
    athletes_missing_birthdates = []
    for _,b in missing_person_bdays_idx:
        athletes_missing_birthdates.append(b)
    unique_missing_athletes = missing_bdays.groupby(by=['Person ID','Competitor','NOC','Event Gender'])['Athlete Birth Date'].count().index
    unique_missing_athletes_cols = ['Athlete_ID','Athlete_Name','Country_Abbv','Athlete_Gender']
    unique_missing_athletes = pd.DataFrame.from_records(unique_missing_athletes,columns=unique_missing_athletes_cols)
    unique_missing_athletes['Athlete_Gender'] = unique_missing_athletes.apply(lambda x: x['Athlete_Gender'].replace('Women','Female'),axis=1)
    unique_missing_athletes['Athlete_Gender'] = unique_missing_athletes.apply(lambda x: x['Athlete_Gender'].replace('Men','Male'),axis=1)
    if write_output:
        unique_missing_athletes.to_csv(SCRAPE_DATA_PATH.joinpath('Missing_Athlete_Birthdays.csv'),index=False)
    else:
        pass
    return unique_missing_athletes



########################################################
# find all missing birthday entires if age and 
# brithdate unknown previously
########################################################
def populate_scraped_missing_birthdates(x,y):
    if (x == 'Unknown') & (y != 'Unknown'):
        return y
    else:
        return x
    

########################################################
# creates new var adj rank which convers dnf to min comp
# position +1
######################################################## 
def Adjust_DNF_Rank(lookup_frame,comp_date,comp_id,event_short,event_gender,rank):
    if rank != -1:
        adjusted_rank = rank
    else:
        adjusted_rank = lookup_frame[(lookup_frame['Competition Date'] == comp_date)&(lookup_frame['Competition ID'] == comp_id)&(lookup_frame['Event Name Short'] == event_short)&(lookup_frame['Event Gender'] == event_gender)].iloc()[0]['Rank'] + 1
    return adjusted_rank



########################################################
# generates a unque key for events
########################################################
def Generate_Event_Hash(date,comp_id,event_name,gender):
    date = str(date).strip()
    comp_id = str(comp_id).strip()
    event_name = str(event_name).strip()
    gender = str(gender).strip()
    hash_tuple = (date,comp_id,event_name,gender)
    hash_out = hash(hash_tuple)
    return hash_out 


########################################################
# returns the number of participants in a comp including dnf
########################################################
def lookup_event_num_participants(lookup_df,event_hash):
    output_frame = lookup_df[lookup_df['Event Hash'] == event_hash]
    return output_frame['Adj_Rank'].iloc()[0]


########################################################
# inverse of min-max scaling function
########################################################
def max_min_scaling(x,x_min,x_max):
    return abs(1-((x-x_min)/(x_max-x_min)))


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