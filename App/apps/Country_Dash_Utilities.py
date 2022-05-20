import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import warnings
from dash import html
import pathlib

from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from joblib import load


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
MODEL_PATH = PATH.joinpath("../Models").resolve()

def Generate_Country_Olympic_Medal_Counts_Bar_Plot(df,event,gender,top_n=15):
    """Generates and returns the olympic medal counts column 
    bar plot used within the country dashboard;

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        event(str): alpine discipline (event name short)
        gender(str): "Men" or "Women
        top_n(int): number of top n countries to inlude in the plot defaults at 15

    Returns:
        dash figure: Olympic medal counts column bar plot
    """


    medal_dict = {
        1:"Gold",
        2:"Silver",
        3:"Bronze"}

    relevent_ranks = [1,2,3]
    df_olympic = df.copy()
    df_olympic = df_olympic[df_olympic['Competition Name']=='Olympic Games']
    df_olympic = df_olympic[['Competition City','Competition Date','Event Hash','Event Gender','Event Name Short','NOC','NOC Name','Adj_Rank']]
    df_olympic = df_olympic[df_olympic['Adj_Rank'].isin(relevent_ranks)]
    title_string = 'Olympic Medals by Country (Top 15)'
    if gender != 'all':
        df_olympic = df_olympic[df_olympic['Event Gender'] == gender]
        title_prefix = gender +"'s" + ' '
    else:
        title_prefix = ''
    if event != 'all':
        df_olympic = df_olympic[df_olympic['Event Name Short'] == event]
        title_prefix = title_prefix + event + ' '
    if title_prefix == '':
        title_prefix = 'Apline '
    title_string = title_prefix + title_string
    df_olympic_grouped = df_olympic.groupby(by=['NOC','NOC Name','Adj_Rank']).count()['Event Hash'].reset_index().rename(columns={'Event Hash':'Medal Count'})
    df_olympic_grouped['Medal Type'] = df_olympic_grouped.apply(lambda x: medal_dict[x['Adj_Rank']],axis=1)
    df_olympic_grouped = df_olympic_grouped.drop(columns=['Adj_Rank'])
    sort_order = list(df_olympic_grouped.groupby(by=['NOC','NOC Name']).sum(['Medal Count']).reset_index().sort_values('Medal Count',ascending=False).iloc()[0:top_n]['NOC Name'])
    df_olympic_grouped = df_olympic_grouped[df_olympic_grouped['NOC Name'].isin(sort_order)]
    bronze_df = df_olympic_grouped[df_olympic_grouped['Medal Type'] == 'Bronze']
    silver_df = df_olympic_grouped[df_olympic_grouped['Medal Type'] == 'Silver']
    gold_df = df_olympic_grouped[df_olympic_grouped['Medal Type'] == 'Gold']
    fig = go.Figure()
    fig.add_trace(go.Bar(x = bronze_df['NOC Name'],y = bronze_df['Medal Count'],name = "Bronze Medals",marker={"color":"#905923"}))
    fig.add_trace(go.Bar(x = silver_df['NOC Name'],y = silver_df['Medal Count'],name = "Silver Medals",marker={"color":"#c0c0c0"}))
    fig.add_trace(go.Bar(x = gold_df['NOC Name'],y = gold_df['Medal Count'],name = "Gold Medals",marker={"color":"#ffd700"}))
    fig.update_layout(barmode='stack',xaxis={'categoryorder':'array', 'categoryarray':sort_order})
    fig.update_layout(xaxis_title='Country',yaxis_title='Medal Count')
    fig.update_layout(title=title_string,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return [fig]

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
    olympic_options_list.insert(0,'Milan Cortina 2026')
    competition_id_list.insert(0,-1)
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
        grouped_athlete_df = grouped_athlete_df[(grouped_athlete_df['Age_Years_At_Olympic_Start']>15.0)&(grouped_athlete_df['Age_Years_At_Olympic_Start']<=60.0)]
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
    df_grouped['Event Gender Pipeline Strength'] = df_grouped.apply(lambda x: 0.14*x['Pre-Peak Age Athletes']+0.53*x['Peak Age Athletes']+0.33*x['Post-Peak Age Athletes'],axis=1)
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
    df_grouped = df_grouped.sort_values(by=['Olympic Year','Overall Pipeline Rank','Event Gender Pipeline Rank'],ascending=False)
    return df_grouped,df_athletes


def Tabulate_Post_Hoc_NOC_Performance(df,competition_ids,competition_names,return_rank=True):
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
    df_copy = df.copy()
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


def Generate_NOC_to_Country_Name_Dict(df):
    """Generates and returns a dictionary for converting noc abbreviations to full
    olympic nation names

    Args:
        df(pd.Dataframe): olympic alpine dataframe

    Returns:
        dict: dictionary mapping noc to noc name
    """
    df_copy = df.copy()
    df_copy = df_copy[['NOC','NOC Name']]
    df_copy = df_copy.drop_duplicates()
    Noc_list = list(df_copy['NOC'])
    Noc_name_list = list(df_copy['NOC Name'])
    name_dict = dict()
    for a,b in zip(Noc_list,Noc_name_list):
        name_dict[a] = b
    return name_dict

def Get_Country_Flag(xref_df,noc_list,className="country-flag"):
    """Returns a coutry flag dash image object

    Args:
        xref_df(pd.Dataframe): ISO_FLAG_XREF dataframe
        noc_list (list): noc countries 
        className (str): css classname desired for the flag

    Returns:
        dash image: returns a given countries flag as a dash image object
    """
    df = xref_df.copy()
    df = df[df['NOC'].isin(noc_list)]
    noc_list = list(set(noc_list))
    flag_dict = dict()
    for i in range(0,len(noc_list)):
        flag_iso_code = np.array(df[df['NOC'] == noc_list[i]]['code'])[0]
        flag_svg_string =  '../assets/flags/' + flag_iso_code + '.svg'
        flag_image_element = html.Img(src=flag_svg_string,className=className)
        flag_dict[noc_list[i]] = flag_image_element
    return flag_dict


def Generate_Olympic_NOC_List(df):
    """Generates a list of all olympic competitor nations

    Args:
        df(pd.Dataframe): olympic alpine dataframe

    Returns:
        list: list of all unique olympic nocs as strings
    """
    df_copy = df.copy()
    df_copy = df_copy[['Competition ID','Competition Name','NOC']]
    df_copy = df_copy[df_copy['Competition Name']=='Olympic Games']
    noc_list = list(np.unique(np.array(df_copy['NOC'])))
    return noc_list


def Generate_NOC_Olympic_Pipeline_History_Chart(df,mode,noc_name):
    """Function generates and returns the olympic hitory boxplot for country detail drilldown section

    Args:
        df(pd.Dataframe): olympic alpine dataframe
        mode (str): determines chart style choice of ['Gender' , 'Event']
        noc_name (str): olympic noc name

    Returns:
        dash figure: returns the olympic pipeline history boxplot
    """
    df_copy = df.copy()
    x = list(df_copy["Olympics Name"])
    x_catagorical_order = list(df_copy[["Olympic Year","Olympics Name"]].sort_values(by='Olympic Year')['Olympics Name'])
    overall_ranks = df_copy[["Olympics Name","Olympic Year","Overall Pipeline Rank"]].drop_duplicates().sort_values(by='Olympic Year')
    fig = go.Figure()
    if mode == 'Gender':
        group_column_name = 'Event Gender'
        chart_title = f'{noc_name} Historic Olympic Pipeline Rankings Grouped by Gender'
    else:
        group_column_name = 'Event Name Short'
        chart_title = f'{noc_name} Historic Olympic Pipeline Rankings Grouped by Event'
    unique_groups= list(df_copy[group_column_name].unique())
    
    for i in range(0,len(unique_groups)):
        df_i = df_copy[df_copy[group_column_name] == unique_groups[i]]
        fig.add_trace(go.Box(x=df_i['Olympics Name'],y=df_i['Event Gender Pipeline Rank'],name=unique_groups[i],legendgroup=unique_groups[i],pointpos=0,boxpoints='all',yaxis='y2',quartilemethod="linear"))
    fig.add_trace(go.Scatter(x=overall_ranks['Olympics Name'],y=overall_ranks['Overall Pipeline Rank'],name='Alpine Pipeline Ranking',mode="lines",line=dict(color="rgba(0,0,0,0.5)",width=2,dash='dash')))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=chart_title,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',boxmode='group',yaxis2=dict(matches='y',layer="above traces",overlaying="y"))
    fig.update_xaxes(categoryorder='array', categoryarray= x_catagorical_order)
    fig.update_xaxes(title_text="Olympic Games")
    fig.update_yaxes(title_text="Pipeline Ranking")
    return fig

def Generate_NOC_Olympic_Pipeline_Comparison_History_Chart(df):
    """Generates and returns the olympic pipeline comparison boxplot for the country comparison dash section

    Args:
        df(pd.Dataframe): olympic alpine dataframe

    Returns:
        dash figure: returns the country comparison boxplot
    """
    df_copy = df.copy()
    x = list(df_copy["Olympics Name"])
    x_catagorical_order = list(df_copy[["Olympic Year","Olympics Name"]].sort_values(by='Olympic Year')['Olympics Name'])
    overall_ranks = df_copy[["Olympics Name","Olympic Year","NOC Name","Overall Pipeline Rank"]].drop_duplicates().sort_values(by=['Olympic Year','NOC Name'])
    fig = go.Figure()
    group_column_name = 'NOC Name'
    unique_groups = list(df_copy[group_column_name].unique())
    chart_title = f'{unique_groups[0]} vs. {unique_groups[1]} Historic Olympic Pipeline Rankings'
    for i in range(0,len(unique_groups)):
        df_i = df_copy[df_copy[group_column_name] == unique_groups[i]]
        ovr_ranks_i = overall_ranks[overall_ranks[group_column_name] == unique_groups[i]]
        fig.add_trace(go.Box(x=df_i['Olympics Name'],y=df_i['Event Gender Pipeline Rank'],name=unique_groups[i],legendgroup=unique_groups[i],pointpos=0,boxpoints='all',yaxis='y2',quartilemethod="linear"))
        overall_ranking_string_i = unique_groups[i] + ' Alpine Pipeline Ranking'
        fig.add_trace(go.Scatter(x=ovr_ranks_i['Olympics Name'],y=ovr_ranks_i['Overall Pipeline Rank'],name=overall_ranking_string_i,mode="lines",line=dict(width=2,dash='dash')))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=chart_title,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',boxmode='group',yaxis2=dict(matches='y',layer="above traces",overlaying="y"))
    fig.update_xaxes(categoryorder='array', categoryarray= x_catagorical_order)
    fig.update_xaxes(title_text="Olympic Games")
    fig.update_yaxes(title_text="Pipeline Ranking")
    return fig


def is_peak_age(age_dict,age,gender,event_name):
    """Catagorizes a given athlete age as pre/post/peak

    Args:
        age_dict(dict): peak age look dictionary
        age (float): athlete age
        gender(str): event gender ['Men' Women']
        event(str): alpine discipline (event name short)
        event_name (str): event name short

    Returns:
        str: returns peak age classification 
    """
    level_1 = age_dict[gender]
    level_2 = level_1[event_name]
    lower_bound = level_2['Age Lower']
    upper_bound = level_2['Age Upper']
    if age > upper_bound:
        output = 'Post-Peak'
    elif age< lower_bound:
        output = 'Pre-Peak'
        
    else:
        output = 'Peak'
    return output

def Get_Event_Gender_Peak_Age_Bounds():
    """Generates a dictionary of the peak age bounds based on the 
    event kde analysis perspective

    Args:
        function takes no arguments

    Returns:
        dict: returns dict containing event and gender paired peak age ranges
    """
    df = pd.read_csv(DATA_PATH.joinpath('Modeling Results/EVENT_KDE_RESULT_DF.csv'))
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

def Scrape_Athlete_Image_Item(athlete_name,athlete_country,athlete_gender,className='dash-image'):
    """Scrapes and returns and FIS profile image from the FIS database

    Args:
        athlete_name (str): athlete full name as string
        athlete_country (str): athlete's noc
        athlete_gender(str): athlete gender ['Men' Women']
        className (str): desired dash class for the image

    Returns:
        dash image: resulting scarped athlete image
    """
    df_xref = pd.read_csv(DATA_PATH.joinpath('Xref/FIS_CompID_XREF.csv'))
    athlete_name_lower = athlete_name.lower()
    df_xref = df_xref[(df_xref['Fullname'] == athlete_name_lower)&(df_xref['Gender'] == athlete_gender)&(df_xref['Nationcode'] == athlete_country)]
    if df_xref.shape[0]>0:
        competitor_id = df_xref.iloc()[0]['Competitorid']
        src_out = f'https://data.fis-ski.com/general/load-competitor-picture/{competitor_id}.html'
        
    else:
        src_out = f'../assets/images/default_fis_no_skiier_image.jpg'
    output_image = html.Img(src=src_out,className=className)
    return output_image



def Generate_Prospect_Table(olympic_name,noc):
    """Generates and returns a table of a given selections olympic prospects

    Args:
        olympic_name(str): olympic names
        noc (str): USOPC country abbv

    Returns:
        dash div element: returns a dash div containing the country's olympic prospect table
    """
    bronze_medal_svg = html.Img(src='../assets/icons/BronzeMedal.svg',className='dash-image wpx25')
    silver_medal_svg = html.Img(src='../assets/icons/SilverMedal.svg',className='dash-image wpx25')
    gold_medal_svg = html.Img(src='../assets/icons/GoldMedal.svg',className='dash-image wpx25')
    pre_age_svg = html.Img(src='../assets/icons/PrePeak.svg',className='dash-image wpx25')
    peak_age_svg = html.Img(src='../assets/icons/Peak.svg',className='dash-image wpx25')
    post_age_svg = html.Img(src='../assets/icons/PostPeak.svg',className='dash-image wpx25')
    peak_age_svg_dict = {'Peak':peak_age_svg,
                        'Post-Peak':post_age_svg,
                        'Pre-Peak':pre_age_svg}
    age_dict = Get_Event_Gender_Peak_Age_Bounds()
    athlete_df = pd.read_csv(DATA_PATH.joinpath('Derived Views/Olympic_Athlete_Data.csv'))
    event_data_df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    athlete_df = athlete_df[(athlete_df['Olympics Name']==olympic_name)&(athlete_df['NOC']==noc)]
    unique_ids = list(athlete_df['Person ID'].unique())
    event_data_df = event_data_df[(event_data_df['NOC']==noc)&(event_data_df['Person ID'].isin(unique_ids))]
    
    current_year = int(dt.date.today().strftime("%Y"))
    oly_year = int(olympic_name.split(' ')[-1])
    oly_date = str(oly_year)+'-01-01'
    if current_year<oly_year:
        
        event_data_wr = event_data_df[(event_data_df['USOC Master Competition Set Name']=='Standing/Ranking List')]
        most_recent_ranking_year = int(event_data_wr['Competition Date'].max().split('-')[0])
        most_recent_ranking_window_start = str(most_recent_ranking_year)+'-01-01'
        most_recent_ranking_window_stop = str(most_recent_ranking_year)+'-12-31'
        event_data_wr = event_data_wr[(event_data_wr['Competition Date']<=most_recent_ranking_window_stop)&((event_data_wr['Competition Date']>=most_recent_ranking_window_start))]
        event_data_wr = event_data_wr.groupby(by=['Person ID','Event Name Short']).min()['Adj_Rank'].reset_index().rename(columns={'Adj_Rank':'World_Rank'})
        
        
    else:
        event_data_wr = event_data_df[(event_data_df['USOC Master Competition Set Name']=='Standing/Ranking List')&(event_data_df['Competition Date']<=oly_date)]
        most_recent_ranking_year = int(event_data_wr['Competition Date'].max().split('-')[0])
        most_recent_ranking_window_start = str(most_recent_ranking_year)+'-01-01'
        most_recent_ranking_window_stop = str(most_recent_ranking_year)+'-12-31'
        event_data_wr = event_data_wr[(event_data_wr['Competition Date']<=most_recent_ranking_window_stop)&((event_data_wr['Competition Date']>=most_recent_ranking_window_start))]
        event_data_wr = event_data_wr.groupby(by=['Person ID','Event Name Short']).min()['Adj_Rank'].reset_index().rename(columns={'Adj_Rank':'World_Rank'})
    
    event_data_df_olympic = event_data_df[(event_data_df['USOC Master Competition Set Name'] == 'Olympic Games')&(event_data_df['Competition Date']<oly_date)]
    past_olympian_list = list(event_data_df_olympic['Person ID'].unique())          
    event_data_df_olympic['Bronze'] = event_data_df_olympic.apply(lambda x: 1 if x['Medal'] == 3 else 0,axis=1)
    event_data_df_olympic['Silver'] = event_data_df_olympic.apply(lambda x: 1 if x['Medal'] == 2 else 0,axis=1)
    event_data_df_olympic['Gold'] = event_data_df_olympic.apply(lambda x: 1 if x['Medal'] == 1 else 0,axis=1)
    event_data_df_olympic = event_data_df_olympic.groupby(by=['Person ID','Event Name Short']).sum()[['Bronze','Silver','Gold']].reset_index()
    athlete_df = athlete_df[['Person ID','Event Gender','Event Name Short','Competitor','Age_Years_At_Olympic_Start']]
    athlete_df['Olympian'] = athlete_df.apply(lambda x: 1 if x['Person ID'] in past_olympian_list else 0,axis=1)
    athlete_wr_df = athlete_df.merge(event_data_wr,how='left',left_on=['Person ID','Event Name Short'],right_on=['Person ID','Event Name Short'])
    athlete_wr_oly = athlete_wr_df.merge(event_data_df_olympic,how='left',left_on=['Person ID','Event Name Short'],right_on=['Person ID','Event Name Short'])
    athlete_wr_oly['World_Rank'] = athlete_wr_oly['World_Rank'].fillna(99).astype(int)
    athlete_wr_oly['Bronze'] = athlete_wr_oly['Bronze'].fillna(0).astype(int)
    athlete_wr_oly['Silver'] = athlete_wr_oly['Silver'].fillna(0).astype(int)
    athlete_wr_oly['Gold'] = athlete_wr_oly['Gold'].fillna(0).astype(int)
    athlete_wr_oly['Bronze'] = athlete_wr_oly.apply(lambda x: html.Div([bronze_medal_svg,x['Bronze']],className='dash-container pl5 shelf ai-center ac-center jc-center'),axis=1)
    athlete_wr_oly['Silver'] = athlete_wr_oly.apply(lambda x: html.Div([silver_medal_svg,x['Silver']],className='dash-container pl5 shelf ai-center ac-center jc-center'),axis=1)
    athlete_wr_oly['Gold'] = athlete_wr_oly.apply(lambda x: html.Div([gold_medal_svg,x['Gold']],className='dash-container pl5 shelf ai-center ac-center jc-center'),axis=1)
    athlete_wr_oly = athlete_wr_oly.sort_values(by=['Event Gender','Event Name Short','Olympian','World_Rank','Competitor'],ascending=[True,True,False,True,True])
    athlete_wr_oly['World_Rank'] = athlete_wr_oly.apply(lambda x: '' if x['World_Rank'] == 99 else x['World_Rank'],axis=1)
    athlete_wr_oly['Age_Years_At_Olympic_Start'] = athlete_wr_oly.apply(lambda x: round(x['Age_Years_At_Olympic_Start'],0),axis=1).astype(int)
    athlete_wr_oly['Age Type'] = athlete_wr_oly.apply(lambda x: is_peak_age(age_dict=age_dict,age=x['Age_Years_At_Olympic_Start'],gender=x['Event Gender'],event_name=x['Event Name Short']),axis=1)
    athlete_wr_oly['Age at Olympic Start'] = athlete_wr_oly.apply(lambda x:html.Div([peak_age_svg_dict[x['Age Type']],x['Age_Years_At_Olympic_Start']],className='dash-container pl5 shelf ai-center ac-center jc-center'),axis=1)
    athlete_wr_oly = athlete_wr_oly.drop(columns=['Age Type','Age_Years_At_Olympic_Start'])
    athlete_wr_oly = athlete_wr_oly.rename(columns={'Gold':'Gold Medals','Bronze':'Bronze Medals','Silver':'Silver Medals','World_Rank':'World Rank','Event Name Short':'Event','Event Gender':'Gender','Competitor':'Name'})
    #uncomment below during demo to activate bio images inline
    athlete_wr_oly['Name'] = athlete_wr_oly.apply(lambda x: html.Div([Scrape_Athlete_Image_Item(x['Name'],noc,x['Gender'],className='dash-image pr5 wpx35'),x['Name']],className='dash-container pl5 shelf ai-center ac-center jc-left'),axis=1)
    athlete_wr_oly = athlete_wr_oly[['Gender','Event','Name','Age at Olympic Start','World Rank','Gold Medals','Silver Medals','Bronze Medals']]
    return athlete_wr_oly

def Generate_Country_Medal_Prediction_Plot(olympic_names,noc,view_mode,rank_mode):
    """Genereates and returns the country medal prediction plot

    Args:
        olympic_names(list of strings): olympic names stored as list of strings
        noc (str): USOPC noc country abbv
        view_mode(str): determines country level vs. olympic level view ['Olympic View' 'Country View']

    Returns:
        dash figure: returns the dash medal predictions plot
    """
    data_df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    age_colors = ['#6baed6','#3182bd','#08519c']
    noc_name_dict = Generate_NOC_to_Country_Name_Dict(data_df)
    ml_data = pd.read_csv(DATA_PATH.joinpath('Derived Views/ML_Input_Data.csv'))
    ml_data['Olympic Year'] = ml_data.apply(lambda x: int(x['Olympics Name'].split(' ')[-1]),axis=1)
    if view_mode == 'Olympic View':
        ml_data = ml_data[ml_data['Olympics Name'].isin(olympic_names)]
        ml_data = ml_data.sort_values(by=['Medal Score'],ascending=False)
        ml_data = ml_data.iloc()[0:15]
    else:
        pass
    ml_data['NOC Name'] = ml_data.apply(lambda x: noc_name_dict[x['NOC']],axis=1)
    lin_reg_model = load(MODEL_PATH.joinpath('Linear Regression/LINREG_optimal_medal_prediction_model.joblib'))
    rt_model = load(MODEL_PATH.joinpath('Regression Trees/RT_optimal_medal_prediction_model.joblib'))
    lasso_model = load(MODEL_PATH.joinpath('Lasso/LASSO_optimal_medal_prediction_model.joblib'))
    ridge_model = load(MODEL_PATH.joinpath('Ridge/RIDGE_optimal_medal_prediction_model.joblib'))
    fig = go.Figure()
    fig2 = go.Figure()
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    X = np.array(ml_data[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']])

    ml_data['LinReg Predicted Medal Scores'] = lin_reg_model.predict(X)
    ml_data['RegressionTree Predicted Medal Scores'] = rt_model.predict(X)
    ml_data['Lasso Predicted Medal Scores'] = lasso_model.predict(X)
    ml_data['Ridge Predicted Medal Scores'] = ridge_model.predict(X)
    
    
    if view_mode == 'Olympic View':
        if rank_mode == 'Medal Score Rank':
            ml_data['Observed Medal Score Rank']= ml_data['Medal Score'].rank(method='min',ascending=False)
            ml_data['LinReg Predicted Medal Score Rank'] = ml_data['LinReg Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['RegressionTree Predicted Medal Score Rank'] = ml_data['RegressionTree Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['Ridge Predicted Medal Score Rank'] = ml_data['Ridge Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['Lasso Predicted Medal Score Rank'] = ml_data['Lasso Predicted Medal Scores'].rank(method='min',ascending=False)
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Observed Medal Score Rank'],name='Observed Rank'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['LinReg Predicted Medal Score Rank'],name='Linear Regression Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['RegressionTree Predicted Medal Score Rank'],name='Regression Tree Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Ridge Predicted Medal Score Rank'],name='Ridge Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Lasso Predicted Medal Score Rank'],name='Lasso Predicted Rank'))
            chart_title = f'Predicted Country Medal Ranks During {olympic_names[0]} Olympics'
            fig.update_layout(title=chart_title)
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Pre-Peak Age Athletes'],name='Pre-Peak Athletes',marker_color=age_colors[0]))
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Peak Age Athletes'],name='Peak Athletes',marker_color=age_colors[1]))
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Post-Peak Age Athletes'],name='Post-Peak Athletes',marker_color=age_colors[2]))
            fig.update_yaxes(title_text="Medal Rank")
            fig2.update_yaxes(title_text="Age Counts")
            fig2.update_xaxes(title_text="Country")
            fig.update_xaxes(title_text="Country")
            fig2.update_layout(title=f'{olympic_names[0]} Pipeline Age Counts')
            

        else:
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Medal Score'],name='Observed Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['LinReg Predicted Medal Scores'],name='Linear Regression Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['RegressionTree Predicted Medal Scores'],name='Regression Tree Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Ridge Predicted Medal Scores'],name='Ridge Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Lasso Predicted Medal Scores'],name='Lasso Predicted Medals Score'))
            chart_title = f'Predicted Country Medal Scores During {olympic_names[0]} Olympics'
            fig.update_layout(title=chart_title)
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Pre-Peak Age Athletes'],name='Pre-Peak Athletes',marker_color=age_colors[0]))
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Peak Age Athletes'],name='Peak Athletes',marker_color=age_colors[1]))
            fig2.add_trace(go.Bar(x=ml_data['NOC Name'],y=ml_data['Post-Peak Age Athletes'],name='Post-Peak Athletes',marker_color=age_colors[2]))
            fig.update_yaxes(title_text="Medal Score")
            fig2.update_yaxes(title_text="Age Counts")
            fig2.update_xaxes(title_text="Country")
            fig.update_xaxes(title_text="Country")
            fig2.update_layout(title=f'{olympic_names[0]} Pipeline Age Counts')
        
    else:
        if rank_mode == 'Medal Score Rank':
            ml_data['Observed Medal Score Rank']= ml_data.groupby(by=['Olympics Name'])['Medal Score'].rank(method='min',ascending=False)
            ml_data['LinReg Predicted Medal Score Rank'] = ml_data.groupby(by=['Olympics Name'])['LinReg Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['RegressionTree Predicted Medal Score Rank'] = ml_data.groupby(by=['Olympics Name'])['RegressionTree Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['Ridge Predicted Medal Score Rank'] = ml_data.groupby(by=['Olympics Name'])['Ridge Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data['Lasso Predicted Medal Score Rank'] = ml_data.groupby(by=['Olympics Name'])['Lasso Predicted Medal Scores'].rank(method='min',ascending=False)
            ml_data = ml_data[ml_data['NOC'].isin(noc)]
            ml_data = ml_data.sort_values(by=['Olympic Year'],ascending=False)
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Observed Medal Score Rank'],name='Observed Rank'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['LinReg Predicted Medal Score Rank'],name='Linear Regression Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['RegressionTree Predicted Medal Score Rank'],name='Regression Tree Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Ridge Predicted Medal Score Rank'],name='Ridge Regression Predicted Rank'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Lasso Predicted Medal Score Rank'],name='Lasso Regression Predicted Rank'))
            chart_title = f'{noc_name_dict[noc[0]]} Predicted Olympic Rank'
            fig.update_layout(title=chart_title)
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Pre-Peak Age Athletes'],name='Pre-Peak Athletes',marker_color=age_colors[0]))
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Peak Age Athletes'],name='Peak Athletes',marker_color=age_colors[1]))
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Post-Peak Age Athletes'],name='Post-Peak Athletes',marker_color=age_colors[2]))
            fig.update_yaxes(title_text="Medal Score")
            fig2.update_yaxes(title_text="Age Counts")
            fig.update_xaxes(title_text="Olympic Games")
            fig2.update_xaxes(title_text="Olympic Games")
            fig2.update_layout(title=f'{noc_name_dict[noc[0]]} Pipeline Age Counts')

        else:
            ml_data = ml_data[ml_data['NOC'].isin(noc)]
            ml_data = ml_data.sort_values(by=['Olympic Year'],ascending=False)
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Medal Score'],name='Observed Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['LinReg Predicted Medal Scores'],name='Linear Regression Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['RegressionTree Predicted Medal Scores'],name='Regression Tree Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Ridge Predicted Medal Scores'],name='Ridge Regression Predicted Medals Score'))
            fig.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Lasso Predicted Medal Scores'],name='Lasso Regression Predicted Medals Score'))
            chart_title = f'{noc_name_dict[noc[0]]} Predicted Olympic Medal Scores'
            fig.update_layout(title=chart_title)
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Pre-Peak Age Athletes'],name='Pre-Peak Athletes',marker_color=age_colors[0]))
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Peak Age Athletes'],name='Peak Athletes',marker_color=age_colors[1]))
            fig2.add_trace(go.Bar(x=ml_data['Olympics Name'],y=ml_data['Post-Peak Age Athletes'],name='Post-Peak Athletes',marker_color=age_colors[2]))
            fig.update_yaxes(title_text="Medal Score")
            fig2.update_yaxes(title_text="Age Counts")
            fig.update_xaxes(title_text="Olympic Games")
            fig2.update_xaxes(title_text="Olympic Games")
            fig2.update_layout(title=f'{noc_name_dict[noc[0]]} Pipeline Age Counts')

    return fig,fig2


