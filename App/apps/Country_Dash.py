import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, clientside_callback, ClientsideFunction
import numpy as np
from apps.Dash_Utilities import *
import pathlib
from apps.Country_Dash_Utilities import *
from plotly.validator_cache import ValidatorCache

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
FLAG_PATH = PATH.joinpath("../assets/flags").resolve()
ICON_PATH = PATH.joinpath("../assets/icons").resolve()
TABLEAU_PATH = PATH.joinpath("../Tableau").resolve()

######################################################################
# Here we read in the neccecary datafiles
# ###################################################################
df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
df = df[df['Class'] == 'Elite']
flag_xref_df = pd.read_csv(DATA_PATH.joinpath('Xref/ISO_NOC_Flag_XREF.csv'))
olympic_ranking_data = pd.read_csv(DATA_PATH.joinpath('Derived Views/Olympic_Ranking_Data.csv'))
olympic_athlete_data = pd.read_csv(DATA_PATH.joinpath('Derived Views/Olympic_Athlete_Data.csv'))

######################################################################
# Here we read the tableau public url from a textfile
# we then add the embedding to an iframe and house it in a container
# ###################################################################
with open(TABLEAU_PATH.joinpath('Tableau_World_Medal_Counts_Embedding_Link.txt')) as f:
    Tableau_Embedding_String = f.readlines()

Tableau_Embedding_String = ' '.join(Tableau_Embedding_String)
tableau_iframe = html.Iframe(srcDoc=Tableau_Embedding_String,style={"height":"850px","width":"1500px","border-color":"transparent","background-color": "transparent"})
tableau_iframe_container = html.Div([tableau_iframe],className='dash-container w100 stack jc-center ai-center',style={'background-color':'transparent'})



######################################################################
# Here we create several convience dictionaries to help our 
# conversion from various abbreviations and full strings
# ###################################################################
NOC_to_Country_Dict = Generate_NOC_to_Country_Name_Dict(df)


olympic_country_noc_list = Generate_Olympic_NOC_List(df)
olympic_country_name_list = [NOC_to_Country_Dict[x] for x in olympic_country_noc_list]
olympic_country_options = [{'label':a,'value':b} for a,b in zip(olympic_country_name_list,olympic_country_noc_list)]

olympics_options = Generate_Olympic_Games_Options(df)
olympic_names_list = [x['label'] for x in olympics_options]
olympic_comp_ids_list = [x['value'] for x in olympics_options]
comp_ids_to_olympic_names = {a:b for a,b in zip(olympic_comp_ids_list,olympic_names_list)}
olympic_names_to_comp_ids = {b:a for a,b in zip(olympic_comp_ids_list,olympic_names_list)}



######################################################################
# Here we manually declare the various dash dropdown options for 
# several differnt common dropdowns used in the dash
# ###################################################################

event_options_all = [
    {'label':'All Events','value':'all'},
    {'label':'Combination','value':'Combination'},
    {'label':'Downhill','value':'Downhill'},
    {'label':'Giant Slalom','value':'Giant Slalom'},
    {'label':'Slalom','value':'Slalom'},
    {'label':'Super G','value':'Super G'},
]

gender_options_all = [
    {'label':'All Genders','value':'all'},
    {'label':'Men','value':'Men'},
    {'label':'Women','value':'Women'}
]

medal_prediction_options = [
    {'label':'Linear Regression','value':'LINREG'},
    {'label':'Ridge','value':'RIDGE'},
    {'label':'Lasso','value':'LASSO'},
    {'label':'Regression Trees','value':'RT'},
]

medal_prediction_view_options = [
    {'label':'Olympic View','value':'Olympic View'},
    {'label':'Country View','value':'Country View'}
]

#distinct from gender_options_all
gender_options =[
    {'label':'Men','value':'Men'},
    {'label':'Women','value':'Women'},
]

#distinct from event_options_all
event_options = [
    {'label':'Combination','value':'Combination'},
    {'label':'Downhill','value':'Downhill'},
    {'label':'Giant Slalom','value':'Giant Slalom'},
    {'label':'Slalom','value':'Slalom'},
    {'label':'Super G','value':'Super G'},
]


######################################################################
# Here we create the full length title shelves for our 5
# primary sections and 2 subtitles for pipline comparison
# ###################################################################

medal_count_chart_title = TitleShelf(title='Olympic Performance History',title_sub_text='')
pipeline_ranking_title = TitleShelf(title='Country Pipeline Rankings', title_sub_text='')
pipeline_comparison_title = TitleShelf(title='Country Pipeline Comparison', title_sub_text='')
pipeline_detail_title = TitleShelf(title='Country Pipeline Detailed', title_sub_text='')
medal_prediction_title = TitleShelf(title='Olympic Medal Score Forecasting', title_sub_text='')

pipeline_ranking_aggregate_title = html.H2('Top Country Pipelines Pre-Olympics')
pipeline_ranking_post_hoc_title = html.H2('Top Country Medals Achieved')

######################################################################
# Here we create the country medal column bar chart
# filtering and downdown options
# ###################################################################

event_filter_drop_down = html.Div([
    html.H2('Filter on Event and Gender'),
    html.Label("Event:"),
    dcc.Dropdown(id="event_filter",
        options=event_options_all ,
        multi=False,
        clearable=False,
        value='all',
        style={"margin-top":"10px","margin-bottom":"10px"}),
    html.Label("Gender:"),
    dcc.Dropdown(id="gender_filter",
        options=gender_options_all,
        multi=False,
        clearable=False,
        value='all',
        style={"margin-top":"10px","margin-bottom":"10px"})   
],className='dash-container stack w10')



######################################################################
# Here we create the pipeline ranking 
# filtering and downdown options
# ###################################################################

country_pipeline_ranking_selections = html.Div([
    html.Div([
    html.Label("Olympics:"),
    dcc.Dropdown(id="pipeline_ranking_olympics_select",
        options=olympics_options,
        multi=False,
        clearable=False,
        value=400941392,
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Genders:"),
    dcc.Dropdown(id="pipeline_ranking_gender_select",
        options=[],
        multi=True,
        clearable=False,
        value=[],
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Events:"),
    dcc.Dropdown(id="pipeline_ranking_events_select",
            options=[],
            multi=True,
            clearable=False,
            value=[],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    ],className='dash-selection-container flex-row')


######################################################################
# Here we create the pipeline comparison section 
# filtering and downdown options
# ###################################################################   

pipeline_comparison_selections_container = html.Div(children=[
    html.Div([html.H3('Comparison Selections:')],className='selection-label-container flex-row'),
    html.Div(children=[
    html.Label("Olympics:"),
    dcc.Dropdown(id="pipeline_comparison_olympics_select",
        options=olympics_options,
        multi=True,
        clearable=False,
        value=[400941392],
        className='dash-dropdown')],
    className='selection-label-container flex-row'),
    html.Div([
    html.Label("Country (Left):"),
    dcc.Dropdown(id="pipeline_comparison_country_select_1",
        options=[],
        multi=False,
        clearable=False,
        value='USA',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Country (Right):"),
    dcc.Dropdown(id="pipeline_comparison_country_select_2",
        options=[],
        multi=False,
        clearable=False,
        value='RUS',
        className='dash-dropdown')],className='selection-label-container flex-row')
    ],className='pipeline-comparisons-selections-container')

######################################################################
# Here we create the pipeline detail section
# filtering and downdown options
# ################################################################### 

pipeline_detail_country_selection = html.Div([
    html.Div([html.H3('Detail Selections:')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Country:"),
    dcc.Dropdown(id="pipeline_detail_country_select",
        options=olympic_country_options,
        multi=False,
        clearable=False,
        value='NOR',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Olympic Games:"),
    dcc.Dropdown(id="pipeline_detail_olympics_select",
        options=[],
        multi=False,
        clearable=False,
        value=400941392,
        className='dash-dropdown')],className='selection-label-container flex-row'),
        
        
],className='dash-container w30 stack jc-left ai-center')

######################################################################
# Here we create the medal prediction section  
# filtering and downdown options
# ################################################################### 

medal_prediction_inputs = html.Div([
    html.Div([dcc.RadioItems(options=['Olympic View','Country View'], value='Olympic View',id ='medal_prediction_view_select')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Olympic Games:"),
    dcc.Dropdown(id="medal_prediction_olympic_select",
        options=olympics_options,
        multi=False,
        clearable=False,
        value=400941392,
        className='dash-dropdown')],className='selection-label-container flex-row',id='medal_prediction_olympics_select_container'),
    html.Div([
    html.Label("Country:"),
    dcc.Dropdown(id ='medal_prediction_country_select',
        options=olympic_country_options,
        multi=False,
        clearable=False,
        value='USA',
        className='dash-dropdown')],className='selection-label-container flex-row',id='medal_prediction_country_select_container'),
    html.Div([dcc.RadioItems(options=['Medal Score Rank','Medal Score'], value='Medal Score',id='medal_prediction_rank_select')],id='medal_prediction_rank_select_container',className='selection-label-container flex-row'),
    
],className='dash-selection-container flex-row')


######################################################################
# Here we create the callback targeted components which will all demonstrate
# loading functionality 
# ################################################################### 

#Medal Counts section dash graph and table components
country_medal_counts_chart = LoadingGraph(graph_id='country_medal_count_plot',className='dash-container shelf w45 h100 ai-center ac-center')


#Pipeline ranking section dash graph and table components
pipeline_ranking_figure = LoadingGraph(graph_id='pipeline_ranking_figure',className='dash-container stack w75')
pipeline_ranking_aggregate_table = dcc.Loading([html.Div([],className='dash-table-container',id='pipeline_ranking_agg_table')],parent_className='dash-table-container')
pipeline_ranking_post_hoc_table = dcc.Loading([html.Div([],className='dash-table-container')],parent_className='dash-table-container',id='pipeline_ranking_post_hoc_table')

#Country pipeline detail section dash graph and table components
pipeline_detail_tables_container = dcc.Loading(html.Div(children=[],id='country_pipeline_detail_table_container',className='dash-container stack w70 jc-center ai-center'),parent_className='dash-container stack w70 jc-center ai-center')
pipeline_detail_plot = LoadingGraph(graph_id='country_pipeline_detail_line_chart',className='dash-container shelf jc-center ai-center w70')

#Country pipeline comparison section dash graph and table components
pipeline_comparison_plot_container = html.Div([LoadingGraph(graph_id='country_pipeline_comparison_chart',className='dash-container stack w70')],className='dash-container w100 ai-center jc-center shelf')
pipeline_comparison_table_container = dcc.Loading(html.Div(children=[],id='country_pipeline_comparison_table_container'),parent_className='country_pipeline_comparison_table_container')

#Medal Prediction section dash graph and table components
medal_prediction_chart = LoadingGraph(graph_id='medal_prediction_primary_chart',className='dash-container stack w70')
medal_prediction_age_chart = LoadingGraph(graph_id='medal_prediction_secondary_chart',className='dash-container stack w70')
medal_prediction_table_container = dcc.Loading(html.Div(children=[],id='medal_prediction_table',className='dash-container stack w70 jc-center ai-center'),parent_className='dash-container stack w70 jc-center ai-center')


###################################################################################### 
# House all of our medal coutns plots and components in a self container shelf
# ################################################################################### 
dash_row_1_container = html.Div(children=[event_filter_drop_down,country_medal_counts_chart],className='dash-container shelf mt25 mb25 ai-start jc-se w100')


###################################################################################### 
# House all of our pipeline ranking plots and components in a self contained shelf
# ################################################################################### 

pipeline_ranking_shelf = html.Div([country_pipeline_ranking_selections,pipeline_ranking_figure],className='dash-container stack jc-center ai-center ac-center')
pipeline_ranking_aggregate_container= html.Div(children=[pipeline_ranking_aggregate_title,pipeline_ranking_aggregate_table],className='dash-table-container')
pipeline_ranking_post_hoc_container  = html.Div(children=[pipeline_ranking_post_hoc_title,pipeline_ranking_post_hoc_table],className='dash-table-container',id='pipeline_ranking_post_hoc_container')
pipeline_ranking_table_container = html.Div([pipeline_ranking_aggregate_container,pipeline_ranking_post_hoc_container],className='dash-container shelf jc-center ai-start')



###################################################################################### 
# House all of our pipeline comparison plots and components in a self container shelf
# ################################################################################### 
pipeline_comparison_title_country_name_container_1 = html.Div(children=[html.H2(children=[],id='country_pipeline_subtitle_1')])
pipeline_comparison_title_country_name_container_2 = html.Div(children=[html.H2(children=[],id='country_pipeline_subtitle_2')])
pipeline_comparison_country_flag_container_1 = html.Div(children=[],id='country_pipeline_comparison_flag_1')
pipeline_comparison_country_flag_container_2 = html.Div(children=[],id='country_pipeline_comparison_flag_2')
pipeline_comparison_container_1 = html.Div(children=[pipeline_comparison_title_country_name_container_1,pipeline_comparison_country_flag_container_1],className='dash-container stack jc-center ai-center w25',style={'text-align':'center'})
pipeline_comparison_container_2 = html.Div(children=[html.H1('Vs.')],className='dash-container stack jc-center ai-center w25',style={'text-align':'center'})
pipeline_comparison_container_3 = html.Div(children=[pipeline_comparison_title_country_name_container_2,pipeline_comparison_country_flag_container_2],className='dash-container stack jc-center ai-center w25',style={'text-align':'center'})
pipeline_comparison_content_container = html.Div(children=[pipeline_comparison_container_1,pipeline_comparison_container_2,pipeline_comparison_container_3],className='dash-container shelf w60 jc-center ai-center')
pipeline_comparison_input_shelf = html.Div([pipeline_comparison_selections_container,pipeline_comparison_content_container],className='dash-container jc-center shelf w100 ai-center')
pipeline_comparison_shelf = html.Div(children=[pipeline_comparison_input_shelf,pipeline_comparison_plot_container,pipeline_comparison_table_container],className='dash-container stack jc-center ai-center ac-center')


###################################################################################### 
# House all of our pipeline detail plots and components in a self container shelf
# ################################################################################### 
pipeline_detail_plot_type_select = dcc.RadioItems(options=['Gender','Event'], value='Gender', id='country_pipeline_detail_line_chart_mode',className='dash-radio-items')
pipeline_detail_country_title_container = html.Div(children=[html.H2(children=[],id='country_pipeline_detail_title')])
pipeline_detail_country_flag_container = html.Div(children=[],id='country_pipeline_detail_flag')
pipeline_detail_country_info_container = html.Div(children=[pipeline_detail_country_title_container,pipeline_detail_country_flag_container],className='dash-container stack')
pipeline_detail_line_chart_container = html.Div([pipeline_detail_plot,pipeline_detail_plot_type_select],className='dash-container w70 shelf jc-center ai-center')
pipeline_detail_lead_conatiner = html.Div([pipeline_detail_country_selection,pipeline_detail_country_info_container],className='dash-container shelf w75 jc-se ai-start')
pipeline_detail_shelf = html.Div([pipeline_detail_lead_conatiner,pipeline_detail_line_chart_container,pipeline_detail_tables_container],className='dash-container stack jc-center ai-center')



###################################################################################### 
# House all of our medal prediction plots and components in a self container shelf
# ################################################################################### 

medal_prediction_input_plot_container = html.Div([medal_prediction_inputs],className='dash-container shelf jc-center ai-start')
medal_chart_1_container = html.Div([medal_prediction_chart],className='dash-container shelf jc-center ai-start')
medal_prediction_secondary_plot_container = html.Div([medal_prediction_age_chart],className='dash-container shelf jc-center ai-start')
medal_prediction_shelf = html.Div([medal_prediction_input_plot_container,medal_chart_1_container,medal_prediction_secondary_plot_container,medal_prediction_table_container])

######################################################################
# Here we take each of our section shelf elements and append them
# and post them to the application's layout in order of appearance
# ################################################################### 

layout = html.Div(children=[
    pipeline_ranking_title,
    pipeline_ranking_shelf,
    pipeline_ranking_table_container,
    pipeline_comparison_title,
    pipeline_comparison_shelf,
    pipeline_detail_title,
    pipeline_detail_shelf,
    medal_prediction_title,
    medal_prediction_shelf,
    medal_count_chart_title,
    dash_row_1_container,
    tableau_iframe_container
])

######################################################################
# Begin Application callback section
# ################################################################### 



######################################################################
# callback updates country medal count plot based on selected filters
# ################################################################### 
@callback(
    [Output(component_id='country_medal_count_plot',component_property='figure')],
    [Input(component_id='event_filter',component_property='value'),
    Input(component_id='gender_filter',component_property='value')]
)
def Update_Country_Medal_Counts_Plot(event,gender):
    df_copy = df.copy()
    fig = Generate_Country_Olympic_Medal_Counts_Bar_Plot(df_copy,event,gender)
    return fig


######################################################################
# callback updates country pipeline ranking plot based on selected filters
# ################################################################### 

@callback(
    [Output(component_id='pipeline_ranking_figure',component_property='figure')],
    [Input(component_id='pipeline_ranking_olympics_select',component_property='value'),
    Input(component_id='pipeline_ranking_gender_select',component_property='value'),
    Input(component_id='pipeline_ranking_events_select',component_property='value')
    ]
)
def Country_Pipeline_Ratings_Bar_Chart(olympic_comp_ids,genders,events):
    olympic_ranking_df = olympic_ranking_data.copy()
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    
    if type(genders) != list:
        genders = [genders]
    
    if type(events) != list:
        events = [events]
    age_colors = ['#6baed6','#3182bd','#08519c']
    olympic_comp_names = [comp_ids_to_olympic_names[x] for x in olympic_comp_ids]
    filtered_df = olympic_ranking_df[olympic_ranking_df['Olympics Name'].isin(olympic_comp_names)]
    filtered_df = olympic_ranking_df[(olympic_ranking_df['Event Gender'].isin(genders))&(olympic_ranking_df['Event Name Short'].isin(events))&(olympic_ranking_df['Olympics Name'].isin(olympic_comp_names))]
    filtered_df['NOC Name'] = filtered_df.apply(lambda x: NOC_to_Country_Dict[x['NOC']],axis=1)
    fig = px.bar(filtered_df, x="NOC Name", y=["Pre-Peak Age Athletes","Peak Age Athletes","Post-Peak Age Athletes"],color_discrete_map={
        'Pre-Peak Age Athletes': age_colors[0],
        'Peak Age Athletes': age_colors[1],
        'Post-Peak Age Athletes':age_colors[2]
    })
    fig.update_layout(title=f'{olympic_comp_names[0]} Age Pipeline Comparison',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(categoryorder="total descending")
    fig.update_xaxes(title_text="Country")
    fig.update_yaxes(title_text="Age Frequencies")
    return [fig]

######################################################################
# callback updates pipeline ranking table based on selected filters
# ################################################################### 
@callback(
    [Output(component_id="pipeline_ranking_agg_table", component_property="children")],
    [Input(component_id='pipeline_ranking_olympics_select',component_property='value'),
    Input(component_id='pipeline_ranking_gender_select',component_property='value'),
    Input(component_id='pipeline_ranking_events_select',component_property='value'),
    ]
)
def Generate_Aggregate_Pipeline_Ranking_Table(olympic_comp_ids,genders,events):
    df_copy = olympic_ranking_data.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    olympic_names = [comp_ids_to_olympic_names[x] for x in olympic_comp_ids]
    df_copy = df_copy[['Olympics Name','Olympic Year','NOC','Event Name Short','Event Gender','Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']]
    df_copy = df_copy[(df_copy['Olympics Name'].isin(olympic_names))&(df_copy['Event Name Short'].isin(events))&(df_copy['Event Gender'].isin(genders))]
    df_copy = df_copy.groupby(by=['Olympics Name','NOC']).sum()[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']].reset_index()
    df_copy['Total NOC Potential Athletes'] = df_copy.apply(lambda x: x['Pre-Peak Age Athletes']+x['Peak Age Athletes']+x['Post-Peak Age Athletes'],axis=1)
    df_copy['Pipeline Strength'] = df_copy.apply(lambda x: 0.25*x['Pre-Peak Age Athletes']+0.5*x['Peak Age Athletes']+0.25*x['Post-Peak Age Athletes'],axis=1)
    df_copy['Overall Pipeline Rank'] = df_copy.groupby(by=['Olympics Name'])['Pipeline Strength'].rank(method='min',ascending=False).astype(int)
    df_copy = df_copy.sort_values(by='Overall Pipeline Rank',ascending=True)
    rank_of_15th_element = df_copy['Overall Pipeline Rank'].iloc[9]
    df_copy = df_copy[df_copy['Overall Pipeline Rank']<=rank_of_15th_element]
    country_flag_element_dict = Get_Country_Flag(flag_xref_df,list(df_copy['NOC'].unique()),className="dash-image wpx40")
    pre_age_svg = html.Img(src='../assets/icons/PrePeak.svg',className='dash-image wpx25')
    peak_age_svg = html.Img(src='../assets/icons/Peak.svg',className='dash-image wpx25')
    post_age_svg = html.Img(src='../assets/icons/PostPeak.svg',className='dash-image wpx25')
    all_age_svg = ''
    output_list = []
    for i in range(0,df_copy.shape[0]):
        noc_i = df_copy['NOC'].iloc()[i]
        rank_i = df_copy['Overall Pipeline Rank'].iloc()[i]
        pre_peak_count_i = df_copy['Pre-Peak Age Athletes'].iloc()[i]
        peak_count_i = df_copy['Peak Age Athletes'].iloc()[i]
        post_peak_count_i = df_copy['Post-Peak Age Athletes'].iloc()[i]
        flag_element_i = country_flag_element_dict[noc_i]
        country_name_i = NOC_to_Country_Dict[noc_i]
        all_athlete_count_i = pre_peak_count_i+peak_count_i+post_peak_count_i
        pre_peak_container_i = html.Div([pre_age_svg,str(pre_peak_count_i)],className='dash-container shelf jc-center ai-center ac-center')
        peak_container_i = html.Div([peak_age_svg,str(peak_count_i)],className='dash-container shelf jc-center ai-center ac-center')
        post_peak_container_i = html.Div([post_age_svg,str(post_peak_count_i)],className='dash-container shelf jc-center ai-center ac-center')
        output_i = [str(rank_i)+')',flag_element_i,country_name_i,pre_peak_container_i,peak_container_i,post_peak_container_i]
        output_list.append(output_i)
    output_table = Array_To_HTML_Table(output_list,classNameRoot='dash-df')
    return [output_table]

######################################################################
# callback updates olympic post-hoc table based on selected filters
# ################################################################### 
@callback(
    [Output(component_id="pipeline_ranking_post_hoc_table", component_property="children")],
    [Input(component_id='pipeline_ranking_olympics_select',component_property='value'),
    Input(component_id='pipeline_ranking_events_select',component_property='value'),
    Input(component_id='pipeline_ranking_gender_select',component_property='value')]
)
def Generate_Post_Hoc_Performance_Table(olympic_comp_ids,events,genders):
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['Event Name Short'].isin(events))&(df_copy['Event Gender'].isin(genders))]
    olympic_comp_names = [comp_ids_to_olympic_names[x] for x in olympic_comp_ids]
    if (len(olympic_comp_ids)==1) and (olympic_comp_ids[0]==-1):
        output = [None]
    else:
        post_hoc_df = Tabulate_Post_Hoc_NOC_Performance(df_copy,competition_ids=olympic_comp_ids,competition_names=olympic_comp_names)
        country_flag_element_dict = Get_Country_Flag(flag_xref_df,list(post_hoc_df['NOC']),className="dash-image wpx40")
        bronze_medal_svg = html.Img(src='../assets/icons/BronzeMedal.svg',className='dash-image wpx25')
        silver_medal_svg = html.Img(src='../assets/icons/SilverMedal.svg',className='dash-image wpx25')
        gold_medal_svg = html.Img(src='../assets/icons/GoldMedal.svg',className='dash-image wpx25')
        all_medal_svg = html.Img(src='../assets/icons/AllMedal.svg',className='dash-image wpx25')
        output_list = []
        for i in range(0,post_hoc_df.shape[0]):
            noc_i = post_hoc_df['NOC'].iloc()[i]
            rank_i = post_hoc_df['Olympic Rank'].iloc()[i]
            bronze_count_i = post_hoc_df['Bronze'].iloc()[i]
            silver_count_i = post_hoc_df['Silver'].iloc()[i]
            gold_count_i = post_hoc_df['Gold'].iloc()[i]
            flag_element_i = country_flag_element_dict[noc_i]
            country_name_i = NOC_to_Country_Dict[noc_i]
            all_medal_count_i = bronze_count_i+silver_count_i+gold_count_i
            bronze_medal_container_i = html.Div([bronze_medal_svg,str(bronze_count_i)],className='dash-container shelf jc-center ai-center ac-center')
            silver_medal_container_i = html.Div([silver_medal_svg,str(silver_count_i)],className='dash-container shelf jc-center ai-center ac-center')
            gold_medal_container_i = html.Div([gold_medal_svg,str(gold_count_i)],className='dash-container shelf jc-center ai-center ac-center')
            all_medal_container_i = html.Div([all_medal_svg,str(all_medal_count_i)],className='dash-container shelf jc-center ai-center ac-center')
            output_i = [str(rank_i)+')',flag_element_i,country_name_i,gold_medal_container_i,silver_medal_container_i,bronze_medal_container_i,all_medal_container_i]
            output_list.append(output_i)
        output = [Array_To_HTML_Table(output_list,classNameRoot='dash-df')]
    return output

###########################################################################################
# callback shows and hides options for poc hoc results for future games that havent occured
# ########################################################################################### 
@callback(
    Output(component_id="pipeline_ranking_post_hoc_container", component_property="className"),
    Input(component_id='pipeline_ranking_olympics_select',component_property='value')
)
def Show_Hide_Post_Hoc_Olympic_Results_For_Future_Olympics(olympic_comp_ids):
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    if (len(olympic_comp_ids)==1) and (olympic_comp_ids[0]==-1):
        table_class_out = 'hidden'
    else:
        table_class_out = 'dash-table-container'
    return table_class_out

###########################################################################################
# callback updates potential gender options/avaliablity based on the given games selected
# sometimes women did not compete especially in early games
# ########################################################################################### 
@callback(
    [Output(component_id='pipeline_ranking_gender_select',component_property='options'),
    Output(component_id='pipeline_ranking_gender_select',component_property='value')],
    [Input(component_id='pipeline_ranking_olympics_select',component_property='value')]
)
def Update_Pipeline_Ranking_Gender_Select_Options(olympic_comp_id):
    df_copy = df.copy()
    if olympic_comp_id == -1:
        unique_genders = ['Men','Women']
    else:
        unique_genders = np.unique(df_copy[df_copy['Competition ID']==olympic_comp_id]['Event Gender'])
    gender_options = [{'label':x,'value':x} for x in unique_genders]
    gender_default_value = gender_options[0]['value']
    return gender_options,gender_default_value

###########################################################################################
# callback updates potential event options/avaliablity based on the given games and gender selected
# ########################################################################################### 
@callback(
    [Output(component_id='pipeline_ranking_events_select',component_property='options'),
    Output(component_id='pipeline_ranking_events_select',component_property='value')],
    [Input(component_id='pipeline_ranking_olympics_select',component_property='value'),
    Input(component_id='pipeline_ranking_gender_select',component_property='value')
    ]
)
def Update_Pipeline_Ranking_Event_Select_Options(olympic_comp_id,genders):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if olympic_comp_id == -1:
        unique_events = ['Downhill','Super G','Giant Slalom','Slalom','Combination']
    else:
        unique_events = np.unique(df_copy[(df_copy['Competition ID']==olympic_comp_id)&(df_copy['Event Gender'].isin(genders))]['Event Name Short'])
    event_options = [{'label':x,'value':x} for x in unique_events]
    event_default_value = event_options[0]['value']
    return event_options,event_default_value


###########################################################################################
# callback updates pipeline comparison country options based on given olympic games
# ########################################################################################### 
@callback(
    [Output(component_id='pipeline_comparison_country_select_1',component_property='options'),
    Output(component_id='pipeline_comparison_country_select_2',component_property='options')],
    Input(component_id='pipeline_comparison_olympics_select',component_property='value')
)
def Update_Pipeline_Comparison_Country_Options(olympic_comp_id):
    df_copy = df.copy()
    if type(olympic_comp_id) != list:
        olympic_comp_id = [olympic_comp_id]
    unique_nocs = list(np.unique(np.array(df_copy[df_copy['Competition ID'].isin(olympic_comp_id)]['NOC'])))
    if -1 in olympic_comp_id:
        future_oly_data = olympic_ranking_data[olympic_ranking_data['Olympics Name'] == 'Milan Cortina 2026']
        future_unique_nocs = list(future_oly_data['NOC'].unique())
        unique_nocs.extend(future_unique_nocs)
        unique_nocs = list(set(unique_nocs))
    noc_names = [NOC_to_Country_Dict[x] for x in unique_nocs]
    country_options = [{'label':a,'value':b} for a,b in zip(noc_names,unique_nocs)]
    value_1 = country_options[0]['value']
    value_2 = country_options[1]['value']
    return country_options,country_options

###########################################################################################
# callback updates comparison country flags based on selections
# ########################################################################################### 
@callback(
    [Output(component_id='country_pipeline_comparison_flag_1',component_property='children'),
    Output(component_id='country_pipeline_subtitle_1',component_property='children'),
    Output(component_id='country_pipeline_comparison_flag_2',component_property='children'),
    Output(component_id='country_pipeline_subtitle_2',component_property='children')],
    [Input(component_id='pipeline_comparison_country_select_1',component_property='value'),
    Input(component_id='pipeline_comparison_country_select_2',component_property='value')]
)
def Update_Pipeline_Comparison_Country_Flags_and_Titles(country_1,country_2):
    title_1 = NOC_to_Country_Dict[country_1]
    title_2 =NOC_to_Country_Dict[country_2]
    country_flag_element_dict = Get_Country_Flag(flag_xref_df,[country_1,country_2],className="dash-image wpx400")
    country_image_1 = country_flag_element_dict[country_1]
    country_image_2 = country_flag_element_dict[country_2]
    return country_image_1,title_1,country_image_2,title_2

###########################################################################################
# callback updates pipeline detail olympic options based on given country selected
# ########################################################################################### 
@callback(
    [Output(component_id='pipeline_detail_olympics_select',component_property='options'),
    Output(component_id='pipeline_detail_olympics_select',component_property='value')],
    Input(component_id='pipeline_detail_country_select',component_property='value'))
def Generate_Pipeline_Detail_Olympic_Options(noc):
    df_copy = df.copy()
    df_olympic = df_copy[['Competition City','Competition Date','Competition ID','USOC Master Competition Set Name','NOC']]
    df_olympic = df_olympic[(df_olympic['USOC Master Competition Set Name'] == 'Olympic Games')&(df_copy['NOC']==noc)]
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
    return olympic_options,olympic_options[0]['value']

###########################################################################################
# callback updates pipeline detail country flag based on selections
# ########################################################################################### 
@callback(
    [Output(component_id='country_pipeline_detail_flag',component_property='children'),
    Output(component_id='country_pipeline_detail_title',component_property='children')],
    Input(component_id='pipeline_detail_country_select',component_property='value')
)
def Update_Pipeline_Detail_Country_Flag_and_Title(noc):
    noc_name = NOC_to_Country_Dict[noc]
    country_flag_element_dict = Get_Country_Flag(flag_xref_df,[noc],className="dash-image wpx400")
    noc_flag = country_flag_element_dict[noc]
    return noc_flag,noc_name

###########################################################################################
# callback updates and generates pipeline comparison table based on selections
# ########################################################################################### 
@callback(
    Output(component_id='country_pipeline_comparison_table_container',component_property='children'),
    [Input(component_id='pipeline_comparison_olympics_select',component_property='value'),
    Input(component_id='pipeline_comparison_country_select_1',component_property='value'),
    Input(component_id='pipeline_comparison_country_select_2',component_property='value'),
    ]
)
def Generate_Country_Pipeline_Comparison_Table(olympic_comp_ids,country_1,country_2):
    df_copy = olympic_ranking_data.copy()
    noc_comp_list = [country_1,country_2]
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    olympic_names = [comp_ids_to_olympic_names[x] for x in olympic_comp_ids]
    df_copy = df_copy[(df_copy['NOC'].isin(noc_comp_list))&(df_copy['Olympics Name'].isin(olympic_names))]
    df_copy = df_copy.sort_values(by=['Olympic Year','NOC','Event Gender','Event Name Short'],ascending=[False,True,True,True])
    df_copy = df_copy.pivot(index=['Olympics Name','Event Gender','Event Name Short'],columns='NOC',values=['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Event Gender Pipeline Rank'])
    df_copy = df_copy.reset_index().fillna(0)
    ind = pd.Index([(x[0] + ' '+ x[1]).rstrip() for x in df_copy.columns.tolist()])
    df_copy.columns = ind
    noc_comparison_index_cols = ['Olympics Name','Event Gender']
    country_flag_element_dict = Get_Country_Flag(flag_xref_df,list(noc_comp_list),className="dash-image wpx20 pr5")
    shared_cols = ['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Event Gender Pipeline Rank']
    for i in range(0,len(shared_cols)):
        col_i = shared_cols[i]
        for j in range(0,len(noc_comp_list)):
            country_j = noc_comp_list[j]
            col_string_ij = col_i+' '+country_j
            country_j_flag_element = country_flag_element_dict[country_j]
            df_copy[col_string_ij] = df_copy.apply(lambda x: html.Div([country_j_flag_element,x[col_string_ij]],className='dash-container pl5 shelf ai-center jc-center ac-center'),axis=1)

    noc_comparison_table_col_names = ['Olympics','Gender','Event Name','Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Event/Gender Pipeline Rank']
    noc_comparison_table_col_spans = [1,1,1,2,2,2,2]
    noc_comparison_table_header = html.Tr(children=[html.Th(children=a,className='dash-df-header',colSpan=b) for a,b in zip(noc_comparison_table_col_names,noc_comparison_table_col_spans)],className='dash-df-header-row')
    noc_comparison_table = Multi_Index_DataFrame_To_HTML_Table(df_copy,index_cols=noc_comparison_index_cols,classNameRoot='dash-df',header=noc_comparison_table_header)
    return [noc_comparison_table]

###########################################################################################
# callback updates and generates pipeline detail table based on selections
# ########################################################################################### 
@callback(
    Output(component_id='country_pipeline_detail_table_container',component_property='children'),
    [Input(component_id='pipeline_detail_olympics_select',component_property='value'),
    Input(component_id='pipeline_detail_country_select',component_property='value'),
    ]
)
def Generate_Country_Pipeline_Detail_Tables(olympic_comp_ids,noc):
    df_copy = olympic_ranking_data.copy()
    if type(olympic_comp_ids) != list:
        olympic_comp_ids = [olympic_comp_ids]
    olympic_names = [comp_ids_to_olympic_names[x] for x in olympic_comp_ids]
    pre_age_svg = html.Img(src='../assets/icons/PrePeak.svg',className='dash-image wpx25')
    peak_age_svg = html.Img(src='../assets/icons/Peak.svg',className='dash-image wpx25')
    post_age_svg = html.Img(src='../assets/icons/PostPeak.svg',className='dash-image wpx25')
    df_copy= df_copy[(df_copy['NOC']==noc)&(df_copy['Olympics Name'].isin(olympic_names))]
    df_copy = df_copy.sort_values(by=['Olympic Year','Event Gender','Event Name Short'],ascending=[False,True,True])
    df_copy = df_copy[['Olympics Name','Overall Pipeline Rank','Event Gender','Event Name Short','Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Total NOC Event Gender Potential_Athletes','Event Gender Pipeline Rank']]
    df_copy['Pre-Peak Age Athletes'] = df_copy.apply(lambda x: html.Div([pre_age_svg,x['Pre-Peak Age Athletes']],className='dash-container pl5 shelf ai-center jc-center ac-center'),axis=1)
    df_copy['Peak Age Athletes'] = df_copy.apply(lambda x: html.Div([peak_age_svg,x['Peak Age Athletes']],className='dash-container pl5 shelf ai-center jc-center ac-center'),axis=1)
    df_copy['Post-Peak Age Athletes'] = df_copy.apply(lambda x: html.Div([post_age_svg,x['Post-Peak Age Athletes']],className='dash-container pl5 shelf ai-center jc-center ac-center'),axis=1)
    df_copy = df_copy.rename(columns={'Event Gender':'Gender','Total NOC Event Gender Potential_Athletes':'Total Potential Athletes','Event Name Short':'Event'})
    noc_ranking_index_cols = [['Olympics Name','Overall Pipeline Rank'],'Gender']
    noc_ranking_table = Multi_Index_DataFrame_To_HTML_Table(df_copy,index_cols=noc_ranking_index_cols,classNameRoot='dash-df')
    noc_detail_athletes = Generate_Prospect_Table(olympic_name=olympic_names[0],noc=noc)
    noc_athlete_index_cols = ['Gender','Event']
    men_noc_detail_athlete_table = Multi_Index_DataFrame_To_HTML_Table(noc_detail_athletes[noc_detail_athletes['Gender']=='Men'],index_cols=noc_athlete_index_cols,classNameRoot='dash-df')
    women_noc_detail_athlete_table = Multi_Index_DataFrame_To_HTML_Table(noc_detail_athletes[noc_detail_athletes['Gender']=='Women'],index_cols=noc_athlete_index_cols,classNameRoot='dash-df')
    prospect_title = html.H2('Olympic Prospects')
    ranking_title = html.H2('Olympic Pipeline Overview')
    table_container = html.Div([men_noc_detail_athlete_table,women_noc_detail_athlete_table],className='dash-container pl5 stack jc-center')
    return [ranking_title,noc_ranking_table,prospect_title,table_container]

###########################################################################################
# callback updates and generates pipeline detail box and line plot based on selections
# ########################################################################################### 
@callback(
    Output(component_id="country_pipeline_detail_line_chart",component_property="figure"),
    [Input(component_id='pipeline_detail_country_select',component_property='value'),
    Input(component_id='country_pipeline_detail_line_chart_mode',component_property='value')]
)
def Generate_Country_Pipeline_Detail_History_Plot(noc,mode):
    df_copy = olympic_ranking_data.copy()
    df_copy = df_copy[df_copy['NOC']==noc]
    noc_name = NOC_to_Country_Dict[noc]
    df_copy = df_copy[['Olympic Year','Olympics Name','Overall Pipeline Rank','Event Gender','Event Name Short','Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Event Gender Pipeline Rank']]
    df_copy = df_copy.sort_values(by=['Olympic Year','Event Gender','Event Name Short'],ascending=[False,True,True])
    fig = Generate_NOC_Olympic_Pipeline_History_Chart(df_copy,mode,noc_name)
    return fig

###########################################################################################
# callback updates and generates pipeline comparison box and line plot based on selections
# ########################################################################################### 
@callback(
    Output(component_id='country_pipeline_comparison_chart',component_property='figure'),
    [Input(component_id='pipeline_comparison_country_select_1',component_property='value'),
    Input(component_id='pipeline_comparison_country_select_2',component_property='value'),
    ]
)
def Generate_Country_Pipeline_Comparison_Box_Plot(noc_1,noc_2):
    df_copy = olympic_ranking_data.copy()
    noc_comparison_list = [noc_1,noc_2]
    df_copy= df_copy[df_copy['NOC'].isin(noc_comparison_list)]
    df_copy['NOC Name'] = df_copy.apply(lambda x: NOC_to_Country_Dict[x['NOC']],axis=1)
    df_copy = df_copy[['Olympic Year','Olympics Name','NOC','NOC Name','Overall Pipeline Rank','Event Gender','Event Name Short','Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes','Event Gender Pipeline Rank']]
    df_copy = df_copy.sort_values(by=['Olympic Year','NOC','Event Gender','Event Name Short'],ascending=[False,True,True,True])
    fig = Generate_NOC_Olympic_Pipeline_Comparison_History_Chart(df_copy)
    return fig

###########################################################################################
# callback updates and generates medal prediction plots
# ########################################################################################### 
@callback(
    [Output(component_id='medal_prediction_primary_chart',component_property='figure'),
    Output(component_id='medal_prediction_secondary_chart',component_property='figure')],
    [Input(component_id='medal_prediction_olympic_select',component_property='value'),
    Input(component_id='medal_prediction_country_select',component_property='value'),
    Input(component_id='medal_prediction_view_select',component_property='value'),
    Input(component_id='medal_prediction_rank_select',component_property='value')]
)
def Update_Medal_Prediction_Chart(olympic_id,noc,view_mode,rank_mode):
    if type(olympic_id) != list:
        olympic_id = [olympic_id]
    if type(noc) != list:
        noc = [noc]
    olympic_names = [comp_ids_to_olympic_names[x] for x in olympic_id]
    fig,fig2 = Generate_Country_Medal_Prediction_Plot(olympic_names,noc,view_mode,rank_mode)
    return fig,fig2

###########################################################################################
# callback hides medal prediction options based off user selections
# ########################################################################################### 
@callback(
    [Output(component_id='medal_prediction_country_select_container',component_property='className'),
    Output(component_id='medal_prediction_olympics_select_container',component_property='className')],
    Input(component_id='medal_prediction_view_select',component_property='value')
)
def Show_Hide_Olympic_Medal_Forecasting_Options(view_mode):
    if view_mode == 'Olympic View':
        oly_dd_class = 'selection-label-container flex-row'
        country_dd_class = 'hidden'

    else:
        oly_dd_class = 'hidden'
        country_dd_class = 'selection-label-container flex-row'
    return country_dd_class,oly_dd_class