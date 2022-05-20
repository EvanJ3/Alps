import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
from apps.Dash_Utilities import *
from apps.Individual_Dash_Utilities import *
from sklearn.neighbors import KernelDensity
import pathlib
import math
import datetime as dt
from apps.Python_Utility_Scripts.Splines import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from plotly.validator_cache import ValidatorCache

##################################
# Create Dash Friendly Data paths
# ################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
FLAG_PATH = PATH.joinpath("../assets/flags").resolve()


################################
# Load page dataframes 
# ###############################
df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
df = df[df['Class'] == 'Elite']
event_df = pd.read_csv(DATA_PATH.joinpath('Derived Views/Timed_Competition_Difficulty.csv'))
flag_xref_df = pd.read_csv(DATA_PATH.joinpath('Xref/ISO_NOC_Flag_XREF.csv'))


############################################
# Create a dictionary mapping all
# Unique person ids to their names
# ###########################################
def Generate_PersonID_to_Name_Index(df):
    df_copy = df[['Person ID','Competitor']]
    df_copy = df_copy.drop_duplicates()
    keys = df['Person ID'].tolist()
    values = df['Competitor'].tolist()
    person_id_dict = dict(zip(keys, values))
    return person_id_dict

Person_ID_Dict = Generate_PersonID_to_Name_Index(df)

######################################################################
# Here we manually declare the various dash dropdown options for 
# several differnt common dropdowns used in the dash as well as color mappings
# ###################################################################

event_color_mapping = {
        "Slalom":"#e41a1c",
        "Giant Slalom":"#377eb8",
        "Super G":"#4daf4a",
        "Downhill":"#984ea3",
        "Combination":"#ff7f00",
        "Overall":"#ffd700"}

kernel_options = [
    {'label':'Gaussian','value':'gaussian'},
    {'label':'Tophat','value':'tophat'},
    {'label':'Epanechnikov','value':'epanechnikov'},
    {'label':'Exponential','value':'exponential'},
    {'label':'Linear','value':'linear'}
]

indv_fit_method_options= [
    {'label':'Parabolic Spline','value':'Parabolic Spline'},
    {'label':'Natural Cubic Spline','value':'Natural Cubic Spline'},
    {'label':'Parabolic B-Spline','value':'Parabolic Basis Spline'},
    {'label':'Natural Cubic B-Spline','value':'Natural Cubic Basis Spline'},
    {'label':'Degree 2 Polynomial Regression','value':'Degree 2 Polynomial Regression'},
    {'label':'Degree 3 Polynomial Regression','value':'Degree 3 Polynomial Regression'},
    {'label':'Degree 4 Polynomial Regression','value':'Degree 4 Polynomial Regression'}
]
##########################################################
## Creates Country Dropdown Options dict for dash input ##
##########################################################

country_dropdown_options = Get_Dash_Country_Options(df)

###############################################
# Create Main Athlete Selection Dropdown Boxes
# #############################################

athlete_selection_components = html.Div(
    children=[
        html.H2('Athlete Selection'),
        html.Label("Country:"),
        dcc.Dropdown(id="select_country",
        options=country_dropdown_options,
        multi=False,
        clearable=False,
        value='USA',
        className='dash-dropdown'
        ),
        html.Label("Gender:"),
        dcc.Dropdown(id="select_gender",
        options=[],
        multi=False,
        clearable=False,
        value='Women',
        className='dash-dropdown'
        ),
        html.Label("Athlete Name:"),
        dcc.Dropdown(id="select_name",
        options=[],
        multi=False,
        clearable=False,
        value='Mikaela Shiffrin',
        className='dash-dropdown'
        )
    ],className='dash-container ml50 stack mt5 mr50 w20 mwpx300'
)


########################################################
# Create Athlete Bio Image,Name, & Country Flag Elements
# ######################################################
athlete_bio_image = html.Img(src='../assets/images/default_athlete_photo.jpg',className='dash-image mhpx130 br75',id='athlete_bio_image')
athlete_bio_name = html.H1([],id='athlete_bio_title')
athlete_country_flag_container = html.Div([],className='dash-container stack jc-center ai-center pl15',id='country_flag_container')


########################################################
# Create Athlete Bio List Fields and Labels
# ######################################################
def Generate_Athlete_Bio_List_Elements():
    label_names = ['Personal ID','Country','Gender','Age','Birthdate','Status','Alias']
    list_components = []
    for i in range(0,len(label_names)):
        label_i = label_names[i]
        label_i_text = label_i + ':'
        label_i_lower = label_i.lower()
        label_i_lower = label_i_lower.replace(' ','_')
        value_i_id = 'athlete_bio_'+label_i_lower+'_select'
        span_label_i = html.Span(label_i_text,className='athlete-bio-list-label')
        span_value_i = html.Span('',className='athlete-bio-list-value',id=value_i_id)
        list_element_i = html.Li([span_label_i,span_value_i])
        list_components.append(list_element_i)
    ol_component = html.Ol(list_components,className='athlete-bio-list')
    return ol_component

athlete_bio_list_components = Generate_Athlete_Bio_List_Elements()

##############################################################
# Join all athlete bio components into a single Bio container
# ############################################################

athlete_bio_components = html.Div([
    html.Div(
        [html.Div([athlete_bio_image]),
        html.Div([athlete_bio_name,athlete_country_flag_container],className='dash-container shelf ai-center jc-center w100',style={'color':'#fdfdfe'}),
        html.Div([],id='athlete_bio_seperator'),
        html.Div([],id='athlete_bio_medal_div',className='dash-container shelf jc-center ai-center pt10 w100',style={'color':'white'})
        ]
    ,className='dash-container p25 jc-center ai-center stack h30 w100',style={'color':'#fdfdfe'}),
    html.Div([athlete_bio_list_components],className='',id='athlete_bio_list_info')
],id='athlete_bio_container')

######################################################################
# Here we create the callback targeted components which will all demonstrate
# loading functionality 
# ################################################################### 

indv_world_ranking_plot = LoadingGraph(graph_id='indv_world_ranking_plot',className='dash-container shelf ai-center ac-center w35')

indv_pie_chart = html.Div([
    dcc.Loading([dcc.Graph(id='event_pie_chart', figure={})],parent_style={"height":"100%", "width":"100%"}),
    dcc.RangeSlider(id='select_season',min=0,max=0,step=1,value=[0,0])
    ],className='dash-container stack jc-center ai-center w20 mt5')


podium_bar_chart = html.Div([
    dcc.Loading([dcc.Graph(id='podium_bar_chart', figure={})],parent_style={"width":"100%"}),
    dcc.RadioItems(options=['All Finishes','Podium Drilldown','Seasonal Performance'], value='All Finishes', id='podium_radio_button',className='dash-container shelf jc-center ai-center ac-center w100',style={'height':'36px'})
    ],className='dash-container stack jc-center ai-center ac-center w50')

difficulty_scatter_plot = LoadingGraph(graph_id='difficulty_scatter_chart', className='dash-container shelf ai-start w30')

world_ranking_kde_plot = LoadingGraph(graph_id='world_ranking_kde_plot', className='dash-container shelf ai-center ac-center w40')

indv_career_plot = LoadingGraph(graph_id='indv_career_plot', className='dash-container w90 shelf ai-center ac-center')

indv_kde_plot = LoadingGraph(graph_id='indv_kde_plot', className='dash-container w40 shelf ai-center ac-center')

##############################################################
# Here we create our dash selection components for the event selection
# filter options as well as kde tuning options
# ############################################################

event_selection_drop_down_1 = html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_event_1",
        options=[],
        multi=False,
        clearable=False,
        value='',
        className='dash-dropdown'),
        html.Label("Kernel:"),
        dcc.Dropdown(id="select_kernel_1",
            options=kernel_options,
            multi=False,
            clearable=False,
            value='gaussian',
            className='dash-dropdown'),
        html.Label("Kernel Bandwidth:"),
        dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom", "always_visible": True},
            id="select_kernel_bandwidth_1",
            className='kernel-slider')
],className='dash-container stack w10')

######################################################################
# Here we create the pipeline ranking 
# filtering and downdown options
# ###################################################################

event_selection_drop_down_2 = html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_event_2",
        options=[],
        multi=False,
        clearable=False,
        value='',
        className='dash-dropdown'),
        html.Label("Kernel:"),
        dcc.Dropdown(id="select_kernel_2",
            options=kernel_options,
            multi=False,
            clearable=False,
            value='gaussian',
            className='dash-dropdown'),
        html.Label("Kernel Bandwidth:"),
        dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom", "always_visible": True},
            id="select_kernel_bandwidth_2",
            className='kernel-slider')
],className='dash-container stack w10')

######################################################################
# Here we create the fit method
# filtering and downdown options
# ###################################################################
fit_selection_drop_down = html.Div([
        html.Label("Fit Method:"),
        dcc.Dropdown(id='select_indv_fit_method',
            options=indv_fit_method_options,
            multi=False,
            clearable=False,
            value='Parabolic Spline',
            className='dash-dropdown'),
        dcc.Checklist(options=[{'label': 'Scaled Rank', 'value': 'Scaled Rank'},],
            value=[],
            id='select_rank_type'
            )
],className='dash-container shelf jc-se ai-center mw75 w100',style={'white-space':'nowrap'})



######################################################################
# create a unique container for the indv carreer plot and selections
# ###################################################################
indv_career_plot_container = html.Div([indv_career_plot,fit_selection_drop_down],className='dash-container stack w40 jc-center ai-center')

##############################################################
# Create and append charts to rowwise dash containers
# ############################################################

dash_row_1_container = html.Div(children=[athlete_selection_components,athlete_bio_components,indv_pie_chart],className='dash-container shelf jc-se ai-start mb25 w100')
dash_row_2_container = html.Div(children=[podium_bar_chart,difficulty_scatter_plot],className='dash-container shelf jc-se ai-center mb25 mt25 w100')
dash_row_4_container = html.Div(children=[event_selection_drop_down_1,world_ranking_kde_plot,indv_world_ranking_plot],className='dash-container shelf mt25 mb25 ai-center jc-se w100')
dash_row_5_container = html.Div(children=[event_selection_drop_down_2,indv_kde_plot,indv_career_plot_container],className='dash-container shelf mt25 mb25 ai-center jc-se w100')

##############################################################
# Append all row dash containers to the page layout
# ############################################################

layout = html.Div(children=[
    dash_row_1_container,
    dash_row_2_container,
    dash_row_4_container,
    dash_row_5_container
])


################################################################
# Adds event callback to update selectable genders in the
# Athlete selection dropdown when a country selection is updated
# ##############################################################

@callback(
    [Output(component_id='select_gender',component_property='options'),Output(component_id='select_gender',component_property='value')],
    [Input(component_id='select_country',component_property='value')]
)
def Get_Gender_Options_For_Selected_Country(selected_country):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['NOC'] == selected_country]
    valid_genders = [{'label':name, 'value':name} for name in sorted(df_copy['Event Gender'].unique())]
    return valid_genders,valid_genders[1]['value']

########################################################################
# Adds event callback to update selectable names in the
# Athlete name selection dropdown when a country and gender are updated
# ######################################################################

@callback(
    Output(component_id='select_name',component_property='options'),Output(component_id='select_name',component_property='value'),
    Input(component_id='select_country',component_property='value'),Input(component_id='select_gender',component_property='value')
)
def Get_Dash_Athlete_Name_Options(selected_country,selected_gender):
    df_copy = df.copy()
    df_copy = df_copy[['Person ID','Competitor','Event Gender','NOC']]
    df_copy = df_copy[(df_copy['NOC'] == selected_country)&(df_copy['Event Gender'] == selected_gender)]
    df_copy = df_copy.sort_values('Competitor')
    unique_id_name_tuples = df_copy.groupby(by=['Person ID','Competitor']).count().index
    names_selected = [{'label':b, 'value':a} for a,b in unique_id_name_tuples]
    return names_selected,names_selected[0]['value']

##########################################################
## Updates the bio athlete title with the 
## corresponding athelte name  
##########################################################

@callback(
    Output(component_id='athlete_bio_title',component_property='children'),
    Input(component_id='select_name',component_property='value')
)
def Update_Athlete_Bio_Name(selected_athlete_id):
    return Person_ID_Dict[selected_athlete_id]

##########################################################
## Adds event callback that updates the selected athletes
##  Difficulty vs. Rank scatter plot
##########################################################
@callback(
    Output(component_id='difficulty_scatter_chart',component_property='figure'),
    Input(component_id='select_name',component_property='value')
)
def Plot_Athlete_Difficulty_Scatterplot(selected_athlete_id):
    df_copy = df.copy()
    event_copy = event_df.copy()
    selected_athlete = Person_ID_Dict[selected_athlete_id]
    fig = Generate_Athlete_Rank_Diff_Scatterplot(df=df_copy,event_df=event_copy,person_id=selected_athlete_id,person_name=selected_athlete)
    return fig

##########################################################
## Searches known FIS database Entries for athlete profile 
# image and updates the corresponding bio image
##########################################################

@callback(
    Output(component_id='athlete_bio_image',component_property='src'),
    [Input(component_id='select_name',component_property='value'),
    Input(component_id='select_country',component_property='value'),
    Input(component_id='select_gender',component_property='value')]
)
def Update_Athlete_Image(selected_athlete_id,selected_country,selected_gender):
    selected_athlete = Person_ID_Dict[selected_athlete_id]
    scrape_results = Scrape_Athlete_Image(selected_athlete,selected_country,selected_gender)
    if (scrape_results is None):
        athlete_image_src = '../assets/images/default_athlete_photo.jpg'
    else:
        athlete_image_src = scrape_results
    return athlete_image_src

##########################################################
## Queries the flag svg database for the athletes corresponding
# country flag and updates the bio flag element  
##########################################################

@callback(
    Output(component_id='country_flag_container',component_property='children'),
    Input(component_id='select_country',component_property='value')
)
def Set_Athlete_Country_Flag(selected_country):
    flag_xref_df_copy = flag_xref_df.copy()
    flag_iso_code = np.array(flag_xref_df_copy[flag_xref_df_copy['NOC'] == selected_country]['code'])[0]
    flag_svg_string =  '../assets/flags/' + flag_iso_code + '.svg'
    flag_image_element = html.Img(src=flag_svg_string,id='athlete_bio_country_flag')
    return flag_image_element

##########################################################
## A series of callbacks that populate the athlete 
## Bio information fields
##########################################################

@callback(
    [Output(component_id='athlete_bio_personal_id_select',component_property='children'),
    Output(component_id='athlete_bio_birthdate_select',component_property='children'),
    Output(component_id='athlete_bio_age_select',component_property='children'),
    Output(component_id='athlete_bio_country_select',component_property='children'),
    Output(component_id='athlete_bio_gender_select',component_property='children'),
    Output(component_id='athlete_bio_status_select',component_property='children'),
    Output(component_id='athlete_bio_alias_select',component_property='children')],
    Input(component_id='select_name',component_property='value')
)
def Update_Athlete_ID_Age_Birthdate_Status_Country_Gender(selected_athlete):
    
    df_copy = df[df['Person ID'] == selected_athlete]
    usopc_id = str(selected_athlete)
    alias = list(df_copy['Athlete Aliases'].dropna().unique())
    if len(alias) == 0:
        alias = '--'
    country = df_copy['NOC'].iloc()[0]
    gender = df_copy['Event Gender'].iloc()[0]
    status_indicator = df_copy['Status'].iloc()[0]
    if status_indicator == 0:
        status = 'Retired'
    else:
        status = 'Active'
    birth_date = df_copy.iloc()[0]['Athlete Birth Date']
    age_years = str(math.floor((dt.datetime.now() - pd.to_datetime(birth_date)).days/365.25))
    return usopc_id,birth_date,age_years,country,gender,status,alias


######################################################################
# Callback updates olympic medal counts in athlete bio based on athelte selected
# ###################################################################
@callback(
    Output(component_id='athlete_bio_medal_div',component_property='children'),
    Input(component_id='select_name',component_property='value')
)
def Update_Athlete_Olympic_Medal_Totals(selected_athlete):
    bronze_medal_svg = html.Img(src='../assets/icons/BronzeMedal.svg',className='dash-image wpx20')
    silver_medal_svg = html.Img(src='../assets/icons/SilverMedal.svg',className='dash-image wpx20')
    gold_medal_svg = html.Img(src='../assets/icons/GoldMedal.svg',className='dash-image wpx20')
    df_copy = df[(df['Competition Name']=='Olympic Games')&(df['Person ID']==selected_athlete)]
    oly_gold_count = df_copy[(df_copy['Medal']==1)].count()['Medal']
    oly_silver_count = df_copy[(df_copy['Medal']==2)].count()['Medal']
    oly_bronze_count = df_copy[(df_copy['Medal']==3)].count()['Medal']
    medal_div = [gold_medal_svg,oly_gold_count,silver_medal_svg,oly_silver_count,bronze_medal_svg,oly_bronze_count]
    return medal_div

##########################################################
## Used to query an athletes career for event groups they
# have competited in 
##########################################################

@callback(
    [Output(component_id='select_event_1',component_property='options'),
    Output(component_id='select_event_1',component_property='value')],
    Input(component_id='select_name',component_property='value')
)
def Get_Athlete_Events1(selected_athlete_id):
    '''
    Removes Combination event from the individual selection as it isn't 
    a world ranking tracked event
    '''
    df_copy = df[['Person ID','Event Name Short']]
    df_copy = df_copy[df_copy['Person ID'] == selected_athlete_id]
    df_copy = df_copy[df_copy['Event Name Short']!= 'Combination']
    event_options = [{'label':name, 'value':name} for name in sorted(df_copy['Event Name Short'].unique())]
    return event_options,event_options[0]['value']

######################################################################
## Used to query an athletes career for event groups they
# have competited in 
# ###################################################################

@callback(
    [Output(component_id='select_event_2',component_property='options'),
    Output(component_id='select_event_2',component_property='value')],
    Input(component_id='select_name',component_property='value')
)
def Get_Athlete_Events2(selected_athlete_id):
    '''
    Removes Overall event from the individual selection as it isn't 
    an individual event
    '''
    df_copy = df[['Person ID','Event Name Short']]
    df_copy = df_copy[df_copy['Person ID'] == selected_athlete_id]
    df_copy = df_copy[df_copy['Event Name Short']!= 'Overall']
    event_options = [{'label':name, 'value':name} for name in sorted(df_copy['Event Name Short'].unique())]
    return event_options,event_options[0]['value']



################################################################
## Creates a line chart of the selected athletes world ranking
## in each sport they compete in over thier entire career  
################################################################
@callback(
    [Output(component_id='indv_world_ranking_plot', component_property='figure')],
    Input(component_id='select_name',component_property='value')
)
def Generate_Athlete_World_Ranking_Plot(selected_athlete_id):
    selected_athlete = Person_ID_Dict[selected_athlete_id]
    df_copy = df[['Season','Event Name Short','Athlete Age Days Derived','Rank' ,'USOC Master Competition Set Name','Person ID']]
    df_copy = df_copy[(df_copy['Person ID'] == selected_athlete_id)&(df_copy['USOC Master Competition Set Name']=='Standing/Ranking List')]
    df_copy = df_copy[['Season','Event Name Short','Athlete Age Days Derived','Rank']]
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    indv_world_rankings = df_copy.groupby(by=['Season','Event Name Short'])['Rank'].mean()
    title_string = f'{selected_athlete} Career World Rankings by Event'
    if indv_world_rankings.shape[0] == 0:
        layout = go.Layout(title=title_string,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(layout=layout)
        fig.update_layout(xaxis_title='Season',yaxis_title='World Ranking')
        no_entries_message = f"No World Rankings for {selected_athlete} Yet"
        fig.update_layout(annotations=[{"text": no_entries_message,
                                        "xref": "paper",
                                        "yref": "paper",
                                        "showarrow": False,
                                        "font": {"size": 18}
        }])
    else:
        rank_array = np.array(indv_world_rankings)
        season_event_array = np.array(indv_world_rankings.index)
        season_array = np.array([a for a,_ in season_event_array])
        event_array = np.array([b for _,b in season_event_array])
        unique_events = np.unique(event_array)
        min_rank = np.min(rank_array)
        max_rank = np.max(rank_array)
        layout = go.Layout(title=title_string)
        fig = go.Figure(layout=layout)
        for i in range(0,unique_events.shape[0]):
            indexes_i = np.array([event_array == unique_events[i]]).squeeze()
            season_array_i = season_array[indexes_i].squeeze()
            event_array_i = event_array[indexes_i].squeeze()
            rank_array_i = rank_array[indexes_i].squeeze()
            fig.add_trace(go.Scatter(x=season_array_i,y=rank_array_i,name=unique_events[i]))
        fig.update_layout(xaxis_title='Season',yaxis_title='World Ranking',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(autorange="reversed")
        fig.update(layout_yaxis_range = [min_rank,max_rank])
        tickvals = np.unique(season_array)
        fig.update_xaxes(tickmode = 'array',tickvals = tickvals)
    return [fig]

##########################################################
## Updates the seasonal range slider for the Season/Event 
## Participation pie chart 
##########################################################
@callback(
    [Output(component_id='select_season',component_property='min'),
    Output(component_id='select_season',component_property='max'),
    Output(component_id='select_season',component_property='value'),
    Output(component_id='select_season',component_property='marks')],
    Input(component_id='select_name', component_property='value')
)
def Generate_Season_Options(selected_athlete_id):
    df_copy = df[['Person ID','Season']]
    df_copy = df_copy[df_copy['Person ID'] == selected_athlete_id]
    season_options = np.array(df_copy['Season'].unique())
    min_season = np.min(season_options)
    max_season = np.max(season_options)
    start_vals = [min_season,max_season]
    season_ticks = np.arange(start=min_season,stop=max_season+1,step=1)
    if season_ticks.shape[0]>10:
        season_ticks = season_ticks[::2]
    else:
        pass
    marks = {int(season_ticks[i]): '{}'.format(str(season_ticks[i])) for i in range(season_ticks.shape[0])}
    step = 1
    return min_season,max_season,start_vals,marks

##########################################################
## Creates and updates the event participation pie chart
# given the selected athlete and the selected season range
# provided by the season options range slider element
##########################################################
@callback(
    [Output(component_id='event_pie_chart',component_property='figure')],
    [Input(component_id='select_name',component_property='value'),
    Input(component_id='select_season',component_property='value')]
)
def Generate_Athlete_Event_Pie_Chart(selected_athlete_id,selected_seasons):
    df_copy = df[['Person ID','USOC Master Competition Set Name','Event Name Short','Season']]
    df_copy = df_copy[(df_copy['Person ID'] == selected_athlete_id)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')]
    if selected_seasons[0] == selected_seasons[1]:
        title_string = str(selected_seasons[0]) + " Season Event Participation"
    else:
        title_string = str(selected_seasons[0]) + ' - ' + str(selected_seasons[1]) + " Seasons Event Participation"
    sel_seasons = np.arange(start=selected_seasons[0],stop=selected_seasons[1]+1,step=1)
    df_copy = df_copy[df_copy['Season'].isin(sel_seasons)]
    events,event_counts = np.unique(df_copy['Event Name Short'],return_counts=True)
    total_events = np.sum(event_counts)
    event_proportions = event_counts/total_events
    color_labels = np.array([event_color_mapping[x] for x in events])
    fig = px.pie(values=event_proportions,names=events)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title=title_string)
    return [fig]

##################################################################
## Updates the podium bar chart according to the athlete selected
## as well as the radio item selected which determines the format
## of the chart view only medals vs. all podiums
##################################################################
@callback(
    [Output(component_id='podium_bar_chart',component_property='figure')],
    [Input(component_id='podium_radio_button',component_property='value'),
    Input(component_id='select_name',component_property='value')]
)
def Generate_Podium_Chart(chart_select,selected_athlete_id):
    selected_athlete = Person_ID_Dict[selected_athlete_id]
    df_copy = df[['Person ID','USOC Master Competition Set Name','Class','Competition Date','Event Name Short','Rank','Athlete Age Days Derived','Season','Adj_Rank']]
    df_copy = df_copy[(df_copy['Person ID'] == selected_athlete_id)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')]
    df_copy = df_copy[['Class','Competition Date','Event Name Short','Rank','Athlete Age Days Derived','Season','Adj_Rank']]
    df_copy.sort_values(by='Competition Date',inplace=True)
    df_copy['Top_1'] = df_copy.apply(lambda x: 1 if x['Rank'] == 1 else 0,axis=1)
    df_copy['Top_2'] = df_copy.apply(lambda x: 1 if x['Rank'] == 2 else 0,axis=1)
    df_copy['Top_3'] = df_copy.apply(lambda x: 1 if x['Rank'] == 3 else 0,axis=1)
    df_copy['Podium'] = df_copy.apply(lambda x: x['Top_1']+x['Top_2']+x['Top_3'],axis=1)
    df_copy['Top_4_10'] = df_copy.apply(lambda x: 1 if (x['Rank'] >=4)&(x['Rank'] <=10) else 0,axis=1)
    df_copy['Top_11_25'] = df_copy.apply(lambda x: 1 if (x['Rank'] >=11)&(x['Rank'] <=30) else 0,axis=1)
    df_copy['Top_26_plus'] = df_copy.apply(lambda x: 1 if (x['Rank'] >=26) else 0,axis=1)
    df_copy['DNF'] = df_copy.apply(lambda x: 1 if (x['Rank'] == -1) else 0,axis=1)
    grouped = df_copy.groupby(by=['Season','Event Name Short']).sum()[['Top_1','Top_2','Top_3','Podium','Top_4_10','Top_11_25','Top_26_plus','DNF']]
    group_index = np.array(grouped.index)
    top_array = np.array(grouped)
    year_array = []
    event_array = []
    fig = go.Figure()
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    for a,b in group_index:
        year_array.append(a)
        event_array.append(b)
    if chart_select == 'Podium Drilldown':
        chart_title_string = f'{selected_athlete} Podium Drilldown'
        fig.update_layout(title_text=chart_title_string)
        fig.update_layout(xaxis_title='Season & Event',yaxis_title='Number of Podiums')
        if np.sum(top_array[:,0:3]) == 0:
            no_entries_message = f"No Podiums for {selected_athlete} Yet"
            fig.update_layout(annotations=[{"text": no_entries_message,
                                            "xref": "paper",
                                            "yref": "paper",
                                            "showarrow": False,
                                            "font": {"size": 28}
        }])
        else:
            fig.add_trace(go.Bar(
            x = [year_array,
                event_array],
            y = top_array[:,2],
            name = "3rd Place Finishes",marker={"color":"#775E39"}
            ))

            fig.add_trace(go.Bar(
            x = [year_array,
                event_array],
            y = top_array[:,1],
            name = "2nd Place Finishes",marker={"color":"#9CB0C0"}
            ))

            fig.add_trace(go.Bar(
            x = [year_array,
                event_array],
            y = top_array[:,0],
            name = "1st Place Finishes",marker={"color":"#E9AF4E"} 
            ))

            
            fig.update_layout(barmode='stack')
            
    elif chart_select == 'All Finishes':
        fig.add_trace(go.Bar(
        x = [year_array,
            event_array],
        y = top_array[:,7],
        name = "DNF",marker={"color":"#969696"}
        ))

        fig.add_trace(go.Bar(
        x = [year_array,
            event_array],
        y = top_array[:,6],
        name = "Position 26+",marker={"color":"#bdd7e7"}
        ))

        fig.add_trace(go.Bar(
        x = [year_array,
            event_array],
        y = top_array[:,5],
        name = "Top 11-25",marker={"color":"#6baed6"}
        ))

        fig.add_trace(go.Bar(
        x = [year_array,
            event_array],
        y = top_array[:,4],
        name = "Top 4-10",marker={"color":"#3182bd"}
        ))

        fig.add_trace(go.Bar(
        x = [year_array,
            event_array],
        y = top_array[:,3],
        name = "Podium Finishes",marker={"color":"#08519c"} 
        ))

        chart_title_string = f'{selected_athlete} Finishes by Year and Sport'
        fig.update_layout(title_text=chart_title_string)
        fig.update_layout(barmode='stack')
        fig.update_layout(xaxis_title='Season',yaxis_title='Total Finishes')

    else:
        df_copy = df_copy[df_copy['Rank']!=-1]
        fig = px.box(df_copy, x="Season", y="Rank", color="Event Name Short")
        fig.update_traces(quartilemethod="exclusive")
        fig.update_yaxes(autorange="reversed",title_text='Finishing Postion')
        fig.update_xaxes(title_text="Season")
        fig.update_layout(title=f'{selected_athlete} Seasonal Performance by Event')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    return [fig]


##########################################################
## Generate the world ranking peak KDE chart 
##########################################################
@callback(
    [Output(component_id='world_ranking_kde_plot',component_property='figure')],
    [Input(component_id='select_name',component_property='value'),
    Input(component_id='select_gender',component_property='value'),
    Input(component_id='select_event_1',component_property='value'),
    Input(component_id='select_kernel_1', component_property='value'),
    Input(component_id='select_kernel_bandwidth_1',component_property='value')]
)
def Generate_WR_KDE_Plot(selected_athlete_id,selected_gender,selected_event,selected_kernel,selected_bw):
    df_copy = df.copy()
    wr_df = Get_Gender_Event_WR(df_copy,gender=selected_gender,event=selected_event,class_name='Elite',post_reg=None)
    wr_indv_df = wr_df[wr_df['Person ID'] == selected_athlete_id]
    wr_indv_df = wr_indv_df.sort_values('Athlete Age Days Derived')
    wr_indv_df = wr_indv_df[wr_indv_df['Rank'] == wr_indv_df['Rank'].min()]
    if wr_indv_df.shape[0]>0:
        indv_best_age = round(wr_indv_df['Athlete Age Days Derived'].iloc()[0]/365.25,2)
        indv_best_rank = wr_indv_df['Rank'].iloc()[0]
    else:
        indv_best_age = None
        indv_best_rank = None
    wr_grouped = wr_df.groupby(by=['Person ID'])['Rank'].min()
    unique_ids = np.array(wr_grouped.index)
    best_rank_achieved = np.array(wr_grouped)
    age_occurance_list = []
    for i in range(0,unique_ids.shape[0]):
        wr_df_copy = wr_df.copy()
        wr_df_copy = wr_df_copy[(wr_df_copy['Person ID'] == unique_ids[i])&(wr_df_copy['Rank'] == best_rank_achieved[i])]
        age_occurance_list.append(wr_df_copy['Athlete Age Days Derived'].mean())
    age_array = np.array(age_occurance_list)/365.25
    mean_age_years = round(np.mean(age_array),2)
    std_age_years = round(np.std(age_array),2)
    min_age = np.min(age_array)
    max_age = np.max(age_array)
    X_plot = np.linspace(min_age-2,max_age+2,200)
    kde = KernelDensity(kernel=selected_kernel,bandwidth=selected_bw).fit(age_array.reshape(-1,1))
    clf_scores_full = np.exp(kde.score_samples(X_plot[:,None]))
    clf_sum_full = np.sum(clf_scores_full)
    samples = Custom_Sample_KDE(kde,1000)
    title_string = f'{selected_gender}\'s {selected_event} Peak World Ranking KDE'
    x_area_trace,y_area_trace = Percentile_Peak_Age_Range_Traces(percentile_range=[.5-(1/6),.5+(1/6)],samples=X_plot,scores=clf_scores_full)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_plot,y=clf_scores_full,name='Density'))
    fig.add_trace(go.Scatter(x=x_area_trace,y=y_area_trace,fillcolor='rgba(112, 122, 244,.6)',fill='tozeroy',line_color='rgba(112, 122, 244,.6)',name='Peak Age Range',mode="none",marker=dict(line=dict(color='rgba(112, 122, 244,.6)',width=2))))
    fig.update_layout(title=title_string,xaxis_title='Age in Years',yaxis_title='Probability Density',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return [fig]

##########################################################
## Generate the individual career performance plot
##########################################################

@callback(
    [Output(component_id='indv_career_plot',component_property='figure')],
    [Input(component_id='select_name',component_property='value'),
    Input(component_id='select_gender',component_property='value'),
    Input(component_id='select_event_2',component_property='value'),
    Input(component_id='select_indv_fit_method',component_property='value'),
    Input(component_id='select_rank_type',component_property='value')]
)
def Generate_Career_Plot(selected_athlete_id,selected_gender,selected_event,fit_method,select_rank_type):
    athlete_name = Person_ID_Dict[selected_athlete_id]
    title_string = f"{athlete_name} {selected_event} Career History"
    fig = go.Figure()
    fig.update_layout(title=title_string,xaxis_title='Age in Years',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    indv_df = Get_Athlete_Timed_Events(df=df,person_id=selected_athlete_id,event=selected_event)
    if 'Scaled Rank' in select_rank_type:
        rank_select_string = 'Scaled Rank'
        fig.update_yaxes(title_text=rank_select_string+' (Higher is Better)')
        Y = np.array(indv_df['Scaled_Rank'])
    else:
        rank_select_string = 'Rank'
        fig.update_yaxes(title_text=rank_select_string+' (Lower is Better)',autorange="reversed")
        Y = np.array(indv_df['Adj_Rank'])
    
    X = np.array(indv_df['Athlete Age Days Derived']/365.25)
    
    fig.add_trace(go.Scatter(x=X,y=Y,name=rank_select_string,mode="markers",marker=dict(color='rgba(53, 95, 246, 0.8)',size=8)))
    if 'Spline' in fit_method:
        if 'Cubic' in fit_method:
            order = 4
        elif 'Parabolic' in fit_method:
            order = 3
        elif 'Piecewise-Linear' in fit_method:
            order = 2
        elif 'Degree 4' in fit_method:
            order = 5
        else:
            pass
        max_knots = 5
        min_knots = 2

        if Y.shape[0] < 8:
            no_entries_message = "Not enough data to interpolate"
            fig.update_layout(annotations=[{"text": no_entries_message,
                                            "xref": "paper",
                                            "yref": "paper",
                                            "showarrow": False,
                                            "font": {"size": 18}}])
        else:
            n_folds = Y.shape[0]-1//2
            if 'Basis' in fit_method:
                spline_output_dict = run_b_spline_cv(X=X,Y=Y,n_folds=n_folds,min_knots=min_knots,max_knots=max_knots,seed=None,order=order)
                best_fit_model = Spline(X=X,Y=Y,knots=spline_output_dict['best_hyperparameter'],order=order)
                best_fit_model.fit()
                best_fit_y_hat = best_fit_model.predict(X)
                legend_name = f" Order {order} Basis Spline with {spline_output_dict['best_hyperparameter']} Knots"
                fig.add_trace(go.Scatter(x=X,y=best_fit_y_hat,name=legend_name,mode="lines",line=dict(color='#e34234',width=2)))
            else:
                spline_output_dict = run_cubic_spline_cv(X=X,Y=Y,n_folds=n_folds,min_knots=min_knots,max_knots=max_knots,seed=None,order=order)
                best_fit_model = Spline(X=X,Y=Y,knots=spline_output_dict['best_hyperparameter'],order=order)
                best_fit_model.fit()
                best_fit_y_hat = best_fit_model.predict(X)
                legend_name = f" Order {order} Spline with {spline_output_dict['best_hyperparameter']} Knots"
                fig.add_trace(go.Scatter(x=X,y=best_fit_y_hat,name=legend_name,mode="lines",line=dict(color='#e34234',width=2)))

    elif 'Polynomial' in fit_method:
        X = X[:,np.newaxis]
        if fit_method == 'Degree 2 Polynomial Regression':
            degree = 2
        elif fit_method == 'Degree 3 Polynomial Regression':
            degree = 3
        elif fit_method == 'Degree 4 Polynomial Regression':
            degree = 4
        else:
            pass
        if Y.shape[0] < 10:
            no_entries_message = "Not enough data to interpolate"
            fig.update_layout(annotations=[{"text": no_entries_message,
                                            "xref": "paper",
                                            "yref": "paper",
                                            "showarrow": False,
                                            "font": {"size": 18}}])
        else:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            clf = LinearRegression().fit(X_poly,Y)
            ypreds = clf.predict(X_poly)
            legend_name = f'Degree {degree} Polynomial Regressor'
            fig.add_trace(go.Scatter(x=X.squeeze(),y=ypreds,name=legend_name,mode="lines",line=dict(color='#e34234',width=2)))
    else:
        pass

    return [fig]

##########################################################
## Generate the individual peak performance KDE plot
##########################################################

@callback(
    [Output(component_id='indv_kde_plot',component_property='figure')],
    [Input(component_id='select_gender',component_property='value'),
    Input(component_id='select_event_2',component_property='value'),
    Input(component_id='select_kernel_2', component_property='value'),
    Input(component_id='select_kernel_bandwidth_2',component_property='value'),
    Input(component_id='select_rank_type',component_property='value')]
)
def Generate_Indv_Single_Event_Peak_Performance_KDE(selected_gender,selected_event,selected_kernel,selected_bandwidth,selected_rank):
    df_copy = df.copy()
    event_df_copy = event_df.copy()
    filtered_timed_events = Get_Gender_Event_Timed_Events(df_copy,selected_gender,event=selected_event,class_name='Elite',include_rankings=False)
    filtered_timed_events = Filter_Timed_Event_Entries(filtered_timed_events,min_entries=6)
    filtered_timed_events = filtered_timed_events.dropna()
    filtered_timed_events = filtered_timed_events[['Person ID','Scaled_Rank','Athlete Age Days Derived','Adj_Rank']]
    if 'Scaled Rank' in selected_rank:
        column_select_string = 'Scaled_Rank'
        rank_type_string = 'Scaled Rank'
    else:
        column_select_string = 'Adj_Rank'
        rank_type_string = 'Rank'
    
    grouped_rank = filtered_timed_events.groupby(by=['Person ID'])[column_select_string].min().reset_index()
    rank_array = np.array(grouped_rank[column_select_string])
    person_id_array = np.array(grouped_rank['Person ID'])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_id_i = person_id_array[i]
        rank_i = rank_array[i]
        athlete_i_df = filtered_timed_events[(filtered_timed_events['Person ID']==person_id_i)&(filtered_timed_events[column_select_string]==rank_i)]
        age_list.append(athlete_i_df['Athlete Age Days Derived'].min())
        

    ages_array = np.array(age_list).reshape(-1, 1)/365.25
    min_age = np.min(ages_array)-3
    max_age = np.max(ages_array)+3
    X_plot = np.linspace(min_age,max_age,100)
    clf = KernelDensity(kernel=selected_kernel,bandwidth=selected_bandwidth).fit(ages_array)
    clf_scores = np.exp(clf.score_samples(X_plot[:,None]))
    fig = go.Figure()
    title_string = f"{selected_gender} {selected_event} {rank_type_string} KDE"
    x_area_trace,y_area_trace = Percentile_Peak_Age_Range_Traces(percentile_range=[.5-(1/6),.5+(1/6)],samples=X_plot,scores=clf_scores)
    fig.add_trace(go.Scatter(x=X_plot,y=clf_scores,name=rank_type_string))
    fig.add_trace(go.Scatter(x=x_area_trace,y=y_area_trace,fillcolor='rgba(112, 122, 244,.6)',fill='tozeroy',line_color='rgba(112, 122, 244,.6)',name='Peak Age Range',mode="none",marker=dict(line=dict(color='rgba(112, 122, 244,.6)',width=2))))
    fig.update_layout(title=title_string,xaxis_title='Age in Years',yaxis_title='Probability Density',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return [fig]