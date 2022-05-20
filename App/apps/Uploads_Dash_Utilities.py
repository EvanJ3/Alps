import pandas as pd
import base64
import datetime
import io
from dash import Dash, dcc, html, dash_table
import numpy as np
import pathlib
from apps.Python_Utility_Scripts.Data_Cleaning_Utility_Functions import *
import os
import math

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
UPLOAD_DATA_PATH = PATH.joinpath("../Data/User Uploaded Data").resolve()
SCRAPE_DATA_PATH = PATH.joinpath("../Data/Scraping Results").resolve()

##############################################################
# manually create the comonly used variables for filtering
# and loading distinct alpine groups and columns
# ############################################################ 
alpine_base_columns = [
    'Class', 'Competition City',
    'Competition Date', 'Competition ID', 'Competition Name',
    'Competition Set Name', 'Competitor', 'Event Gender', 'Event ID',
    'Event Name', 'Event Name Short', 'Medal',
    'NOC', 'NOC Name', 'Person ID', 'Result', 'Result Status',
    'Result Type', 'Season', 'Team Mbrs',
    'Team Member Select', 'Tied',  'USOC Competition Set Name',
    'USOC Master Competition Set Name', 'Person Age Days',
    'Person Age Years', 'Rank'
]

na_values = ['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']

solo_mens_olympic_events = ['Slalom - Men','Giant Slalom - Men','Super G - Men','Downhill - Men','Combination - Men','Overall - Men']
solo_womens_olympic_events = ['Slalom - Women','Giant Slalom - Women','Super G - Women','Downhill - Women', 'Combination - Women','Overall - Women']
solo_mens_non_olympic_events = ['Parallel Slalom - Men','Parallel Giant Slalom - Men','KO Slalom - Men','Parallel - Men','Super Combined - Men']
solo_womens_non_olympic_events = ['Parallel Slalom - Women','Parallel Giant Slalom - Women','KO Slalom - Women','Parallel - Women','Super Combined - Women']
team_mixed_olympic_events = ['Team - Mixed']
team_mixed_non_olympic_events = ['Parallel Team - Mixed']

team_olympic_events = team_mixed_olympic_events
team_non_olympic_events = team_mixed_non_olympic_events
solo_olympic_events = solo_mens_olympic_events + solo_womens_olympic_events
solo_non_olympic_events = solo_mens_non_olympic_events + solo_womens_non_olympic_events
all_team_events = team_olympic_events+team_non_olympic_events
all_mens_events = solo_mens_olympic_events+solo_mens_non_olympic_events
all_womens_events = solo_womens_olympic_events + solo_womens_non_olympic_events

solo_event_irrelevent_columns = ['Team Mbrs','Team Member Select']
team_event_irrelevent_columns = ['Person ID']

##############################################################
# Cleans the concatenated upload and base dataset from scratch
# ############################################################ 

def Clean_Alpine_Data():
    #load df
    print('Entered Cleaning Phase')
    df = pd.read_csv(DATA_PATH.joinpath('AlpineSkiingConcat.csv'),dtype=str,keep_default_na=False,na_values=na_values)
    print('Successfully Read in Concatenated Data')
    df = df[df['Info Strada Sport Name']=='Alpine Skiing']
    print('Successfully Filtered Alpine Events')
    df = df[alpine_base_columns]
    print('Successfully Removed Un-used Columns')
    #combine combined and super combined events
    df['Event Name Short'] = df.apply(lambda x:'Combination' if x['Event Name Short']=='Super Combined' else x['Event Name Short'],axis=1)
    print('Successfully Merged Super Combined and Combined Entries in Event Name Short Column')
    df['Event Name'] = df.apply(lambda x:'Combination - Men' if x['Event Name']=='Super Combined - Men' else x['Event Name'],axis=1)
    df['Event Name'] = df.apply(lambda x:'Combination - Women' if x['Event Name']=='Super Combined - Women' else x['Event Name'],axis=1)
    print('Successfully Merged Super Combined and Combined Entries in Event Name Column')
    #converts missing competition city entries to 'Unknown'
    df['Competition City'] = df.apply(lambda x: convert_empty_strings(x['Competition City'],replacement_string='Unknown'),axis=1)
    #converts MM-DD-YYYY into pandas date format YYYY-MM-DD-HH:MM:SS:NS
    df['Competition Date'] = pd.to_datetime(df['Competition Date'], infer_datetime_format=True)
    #converts competition to integer encoding had no blanks
    df['Competition ID'] = df['Competition ID'].astype(float).astype(np.int32)
    #converts blank competiton set names to "Other"
    df['Competition Set Name'] = df.apply(lambda x: convert_empty_strings(x['Competition Set Name'],replacement_string='Other'),axis=1)
    #replace blank entries in the competitors columns with "Unknown"
    df['Competitor'] = df.apply(lambda x: convert_empty_strings(x['Competitor'],replacement_string='Unknown'),axis=1)
    #converts Event ID into integer
    df['Event ID'] = df['Event ID'].astype(float).astype(np.int32)
    #convert medal to integers 0=no medal; 3=bronze; 2=silver; 1=gold;
    df['Medal'] =  df.apply(lambda x: convert_medal_to_integer(x['Medal']),axis=1)
    #converts person ID blanks and -1 entires to -1 integer all were previously used to identify team events with no one person id
    df['Person ID'] = df.apply(lambda x: convert_empty_strings(x['Person ID'],replacement_string='-1'),axis=1)
    df['Person ID'] = df['Person ID'].astype(float).astype(np.int32)
    df = clean_missing_result_entries(df)
    df['Result Status'] = df.apply(lambda x: convert_empty_strings(x['Result Status'],replacement_string='FIN'),axis=1)
    #combines the split season format to the single year convention
    df = combine_split_seasons(df)
    df['Season'] = df['Season'].astype(float).astype(np.int32)
    #converts blank team member entries for solo comps into "NA"
    df['Team Mbrs'] = df.apply(lambda x: convert_empty_strings(x['Team Mbrs'],replacement_string='N/A'),axis=1)
    #converts a team member select into a binary indiciator variable
    df['Team Member Select'] = df.apply(lambda x: x['Team Member Select'].replace('Yes','1'),axis=1)
    df['Team Member Select'] = df.apply(lambda x: x['Team Member Select'].replace('No','0'),axis=1)
    df['Team Member Select'] = df['Team Member Select'].astype(float).astype(np.int32)
    #converts the tied variable into an integer indicator variable 1 for a tie 0 for no tie
    df['Tied'] = df.apply(lambda x: convert_empty_strings(x['Tied'],replacement_string='0'),axis=1)
    df['Tied'] = df.apply(lambda x: x['Tied'].replace('TIE','1'),axis=1)
    df['Tied'] = df['Tied'].astype(float).astype(np.int32)
    #add string to represent blanks as unknown dates
    df['Person Age Days'] = df.apply(lambda x: convert_empty_strings(x['Person Age Days'],replacement_string='Unknown'),axis=1)
    df['Person Age Years'] = df.apply(lambda x: convert_empty_strings(x['Person Age Years'],replacement_string='Unknown'),axis=1)
    df['Person Age Days'] = df.apply(lambda x: exact_replace_entry(x['Person Age Days'],'0','Unknown'),axis=1)
    #Remove rank entries for withdrawn athletes
    df = df[df['Result Status'] != 'WDW']
    #Change retiered result entries to conform with DNF
    df['Result Status'] = df.apply(lambda x: x['Result Status'].replace('RET','DNF'),axis=1)
    #drop entires where rank and result status are unknown or missing 
    df = df[~((df['Result Status'] == 'Unknown')&(df['Rank'] == ''))]
    #Change DNS/DNS to DNF
    df['Result Status'] = df.apply(lambda x: x['Result Status'].replace('DNF/DNS','DNF'),axis=1)
    #Drop DNQ result status
    df = df[df['Result Status'] != 'DNQ']
    #Code -1 for rank where DNF,DNS,DSQ
    df['Rank'] = df.apply(lambda x: rank_dnf_formater(x['Result Status'],x['Rank']),axis=1)
    #Convert "Unknown" result status entries to "FIN" if rank present
    df['Result Status'] = df.apply(lambda x: format_missing_result_status(x['Result Status'],x['Rank']),axis=1)
    df['Rank'] = df['Rank'].astype(float).astype(np.int32)
    #Converts standing and ranking list entries to have N/a as city entry instead of Unknown to mainatin consisientcy
    df['Competition City'] = df.apply(lambda x: format_world_ranking_city_entires(x['USOC Master Competition Set Name'],x['Competition City']),axis=1)
    #fills NAT entires for compitiion date to 'Unknown'
    df['Competition Date'] = df['Competition Date'].fillna('Unknown')
    print('Successfully Filled Missing Entries')
    #mens and womens overall compeition rankings from 1989-1967 have no date of measurement.
    #this function assumes the standard december 1 of that season measurement date and repalces these entries
    df['Competition Date'] = df.apply(lambda x: correct_missing_comp_dates(x['Competition Date'],x['Season']),axis=1)
    print('Successfully Reconciled Missing Competition Dates')
    #Remove entries where we have no person ID
    df = df[df['Person ID'] != -1]
    print('Starting Athlete Alias Identification Process')
    df['Athlete Aliases'] = df.apply(lambda x: Return_Alias_Names_From_Person_ID(df,x['Person ID']),axis=1)
    print('Generating Athlete Alias Dictionary')
    person_id_to_most_recent_name_dict = Return_Most_Recent_Athlete_Name_Dict(df)
    print('Adding Athlete Aliases')
    df['Competitor'] = df.apply(lambda x: person_id_to_most_recent_name_dict[x['Person ID']],axis=1)
    print('Successfully Added Athlete Aliases')
    solo_events_df = df[df['Event Name'].isin(solo_olympic_events)]
    solo_events_df['Athlete Birth Date'] = solo_events_df.apply(lambda x: age_days_to_birthday(x['Competition Date'],x['Person Age Days']),axis=1)
    solo_events_df = reconcile_multiple_athlete_birthdays(solo_events_df)
    print('Writing Missing Athlete Entries to Scraper')
    unique_missing_athletes = find_missing_athletes(solo_events_df,write_output=True)
    print('Successfuly Wrote Missing Athlete Entries to Scraper')
    scraped_results_frame = pd.read_csv(SCRAPE_DATA_PATH.joinpath('Alpine_Skiing_Scraping_Entries_to_Merge.csv'))
    print('Successfully Read Data Scraping Results Frame')
    solo_events_df_merged = solo_events_df.merge(scraped_results_frame, how='left', left_on='Person ID', right_on='Athlete_ID')
    print('Successfully Merged Data Scraping Results Frame')
    solo_events_df_merged['Athlete Scraped Birthdays'].fillna('Unknown',inplace=True)

    solo_events_df_merged['Athlete Birth Date'] = solo_events_df_merged.apply(lambda x: populate_scraped_missing_birthdates(x['Athlete Birth Date'],x['Athlete Scraped Birthdays']),axis=1)

    solo_events_df_merged.drop(columns=['Athlete_ID','Athlete Scraped Birthdays'],inplace=True)
    solo_events_df_merged = solo_events_df
    solo_events_df = solo_events_df[solo_events_df['Athlete Birth Date'] != 'Unknown']
    solo_events_df['Athlete Birth Date'] = pd.to_datetime(solo_events_df['Athlete Birth Date'], infer_datetime_format=True)
    solo_events_df['Athlete Age Days Derived'] = solo_events_df.apply(lambda x: (x['Competition Date'] - x['Athlete Birth Date']).days,axis=1)
    solo_events_df['Athlete Age Days Derived'] = solo_events_df['Athlete Age Days Derived'].astype(float).astype(np.int32)
    solo_events_df = solo_events_df.drop(columns=['Person Age Years','Person Age Days','Team Mbrs','Team Member Select'])
    solo_events_df = solo_events_df.drop_duplicates()

    grouped_frame = solo_events_df.groupby(by=['Competition Date','Competition ID','Event Name Short','Event Gender']).max(['Rank'])
    grouped_frame.reset_index(inplace=True)
    solo_events_df['Adj_Rank'] = solo_events_df.apply(lambda x: Adjust_DNF_Rank(grouped_frame,x['Competition Date'],x['Competition ID'],x['Event Name Short'],x['Event Gender'],x['Rank']),axis=1)
    print('Checking for Retired Athletes')
    unique_person_ids = np.array(solo_events_df['Person ID'].unique())
    status_dict = dict()
    most_recent_event = solo_events_df.sort_values(by='Competition Date').iloc()[-1]['Competition Date']
    for i in range(0,unique_person_ids.shape[0]):
        df_copy = solo_events_df.copy()
        df_copy = df_copy[df_copy['Person ID'] == unique_person_ids[i]]
        last_comp_date = df_copy.sort_values(by='Competition Date').iloc()[-1]['Competition Date']
        years_since_last_comp = (most_recent_event - pd.to_datetime(last_comp_date)).days/365.25
        if years_since_last_comp >= 1.1:
            status = 0
        else:
            status = 1

        status_dict[unique_person_ids[i]] = status

    solo_events_df['Status'] = solo_events_df.apply(lambda x: status_dict[x['Person ID']],axis=1)
    print('Successfully Updated Retirment Status of Athletes')
    solo_events_df['Event Hash'] = solo_events_df.apply(lambda x: Generate_Event_Hash(x['Competition Date'],x['Competition ID'],x['Event Name Short'],x['Event Gender']),axis=1)
    event_max_entrants = solo_events_df.groupby(by=['Event Hash'])['Adj_Rank'].max().reset_index()
    solo_events_df['Scaled_Rank'] = solo_events_df.apply(lambda x: max_min_scaling(x['Adj_Rank'],1,lookup_event_num_participants(lookup_df=event_max_entrants,event_hash=x['Event Hash'])+1),axis=1)
    print('Successfully Completed Cleaning Transformations')
    print('Begin Writing Cleaned Data')
    solo_events_df.to_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'),index=False,encoding='utf-8')
    print('Sucessfully Completed Cleaned Data Write')
    os.remove(DATA_PATH.joinpath('AlpineSkiingConcat.csv'))
    print('Successfully Removed Temporary Files')


##############################################################
# Concatenates new and base alpine datasets
# ############################################################ 
def Concatenate_New_Alpine_Data(df_new):
    df_og = pd.read_csv(DATA_PATH.joinpath('AlpineSkiing.csv'),dtype=str,keep_default_na=False,na_values=na_values)
    df_og_cols = list(df_og.columns)
    df_new_cols = list(df_new.columns)
    col_names_same = True
    for i in range(0,len(df_og_cols)):
        if df_og_cols[i] in df_new_cols:
            pass
        else:
            col_names_same = False
            break
    if col_names_same:
        df_concat = pd.concat([df_og, df_new])
        exit_status = 1
    else:
        df_concat = None
        exit_status = 0
    return df_concat,exit_status

##############################################################
# top level function that calls and monitors the cleaning
# and concatenation process updates logs accordingly
# ############################################################ 
def Concatenate_and_Clean_New_Data(df_new):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    df_concat,exit_status = Concatenate_New_Alpine_Data(df_new)
    if exit_status == 1:
        print('Concatenation Sucessful')
        print('Writing Concatenated Data')
        df_concat.to_csv(DATA_PATH.joinpath('AlpineSkiingConcat.csv'),index=False)
        print('Successfully Wrote Concatenated Data')
        print('Beginning Cleaning of Concatenated Data')
        Clean_Alpine_Data()
        try:
            Generate_Competition_Difficulty_Table()
            df_concat.to_csv(DATA_PATH.joinpath('AlpineSkiing.csv'),index=False)
        except:
            raise Exception("Data Uploaded Successfully, but there was a problem re-running the competition difficulty statistics")
    else:
        raise Exception("Error in data concatenation, data columns are not in proper format please see the example file for proper formatting convention")

##############################################################
# Cleans the concatenated upload and base dataset from scratch
# ############################################################ 
def scale_comp_difficulty(lookup_df,event,gender,class_level,value):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    reduced_lookup_df = lookup_df[(lookup_df['Event Name Short']==event)&(lookup_df['Event Gender']==gender)&(lookup_df['Class']==class_level)]
    max_val =reduced_lookup_df['max_difficulty'].iloc()[0]
    min_val =reduced_lookup_df['min_difficulty'].iloc()[0]
    scaled_val = min_max_scaling(value,min_val,max_val)
    return scaled_val

##############################################################
# Updates the timed competition table based on new 
# data uploads
# ############################################################
def Generate_Competition_Difficulty_Table():
    print('Beginning Update of Timed Competitions Data Table')
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
    print('Beginning Writing of Updated Timed COmpetitions Data Table')
    timed_comp_df.to_csv(DATA_PATH.joinpath('Derived Views/Timed_Competition_Difficulty.csv'),index=False)
    print('Successfully Updated Timed Competitions Data Table')

##############################################################
# Somewhat sterilizes the user input file names to a time
# stamped filename
# ############################################################
def sterilize_filename(filename):
    clean_filename = filename.replace('.csv','')
    clean_filename = clean_filename.replace('.xls','')
    clean_filename.lower()
    clean_filename = clean_filename.replace('-','_')
    clean_filename = clean_filename.replace(' ','_')
    clean_filename = clean_filename.replace(':','_')
    clean_filename = clean_filename.replace(';','')
    return clean_filename

##############################################################
# parse the file contents of user uploads and calls lower level
# cleaning, concatenation , and update functions as needed
# ############################################################
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            upload_date_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            upload_date_string = upload_date_string.replace(':','_')
            upload_date_string = upload_date_string.replace('-','_')
            filename_cleaned = sterilize_filename(filename)
            file_name = 'User_Upload_'+filename_cleaned+'_'+upload_date_string+'.csv'
            print('Reading User Uploaded Data')
            df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            print('Successfully Read User Uploaded Data')
            print('Writing User Uploaded Data')
            df_new.to_csv(UPLOAD_DATA_PATH.joinpath(file_name),index=False,encoding='utf-8')
            print('Successfully Wrote User Uploaded Data')
            print('Beginning Cleaning and Concatenation of User Data and Exisiting Data')
            Concatenate_and_Clean_New_Data(df_new)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            upload_date_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            upload_date_string = upload_date_string.replace(':','_')
            upload_date_string = upload_date_string.replace('-','_')
            filename_cleaned = sterilize_filename(filename)
            file_name = 'User_Upload_'+filename_cleaned+'_'+upload_date_string+'.csv'
            print('Reading User Uploaded Data')
            df_new = pd.read_excel(io.BytesIO(decoded))
            print('Successfully Read User Uploaded Data')
            print('Writing User Uploaded Data')
            df_new.to_csv(UPLOAD_DATA_PATH.joinpath(file_name),index=False,encoding='utf-8')
            print('Successfully Wrote User Uploaded Data')
            print('Beginning Cleaning and Concatenation of User Data and Exisiting Data')
            Concatenate_and_Clean_New_Data(df_new)
            
    except Exception as e:
        print(e)
        os.remove(UPLOAD_DATA_PATHjoinpath(file_name))
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename+' Successfully Uploaded'),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Hr()
    ])