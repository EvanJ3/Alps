import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback,State
import numpy as np
from apps.Dash_Utilities import *
from apps.Event_Dash_Utilities import *
import pathlib
from sklearn.neighbors import KernelDensity
from plotly.validator_cache import ValidatorCache
import time

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
FLAG_PATH = PATH.joinpath("../assets/flags").resolve()

######################################################################
# Here we read in the neccecary datafiles
# ###################################################################
df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
df = df[df['Class'] == 'Elite']
event_df = pd.read_csv(DATA_PATH.joinpath('Derived Views/Timed_Competition_Difficulty.csv'))

######################################################################
# Here we manually declare the various dash dropdown options for 
# several differnt common dropdowns used in the dash
# ###################################################################
peak_age_method_options = [
    {'label':'Standard Deviation','value':'std'},
    {'label':'Percent Under Curve','value':'percentile'},
    {'label':'Custom Selection','value':'custom'}
]


event_options = [
    {'label':'Combination','value':'Combination'},
    {'label':'Downhill','value':'Downhill'},
    {'label':'Giant Slalom','value':'Giant Slalom'},
    {'label':'Slalom','value':'Slalom'},
    {'label':'Super G','value':'Super G'},
]

kde_plot_type_options =[
    {'label':'Probability Density Plot','value':'pdf'},
    {'label':'Cumulative Density Plot','value':'cdf'},
    {'label':'Probability Difference Plot','value':'delta'}
]


kernel_options = [
    {'label':'Gaussian','value':'gaussian'},
    {'label':'Tophat','value':'tophat'},
    {'label':'Epanechnikov','value':'epanechnikov'},
    {'label':'Exponential','value':'exponential'},
    {'label':'Linear','value':'linear'}
]

kde_options = [
    {'label':'Rank View','value':'indv_rank_kde'},
    {'label':'Event View','value':'event_kde'},
    {'label':'World Rank View','value':'indv_wr_kde'},
    {'label':'Regulation View','value':'reg_kde'}
]

gender_options =[
    {'label':'Men','value':'Men'},
    {'label':'Women','value':'Women'},
]

violin_plot_type_options = [
    {'label':'Event View','value':'event_view'},
    {'label':'Rank View','value':'indv_rank_view'},
    {'label':'World Rank View','value':'indv_wr_view'},
    {'label':'Regulation View','value':'reg_view'}
]

age_range_options = [
    {'label':'Event Level','value':1},
    {'label':'Peak Individual Rank','value':2},
    {'label':'Peak Individual World Rank','value':3}
]

######################################################################
# Here we create the primary kde plot chart
# filtering and downdown options
# ###################################################################

primary_kde_selections_top = html.Div([
    html.Div([
    html.Label("KDE Type:"),
    dcc.Dropdown(id="select_primary_kde_method",
        options=kde_options,
        multi=False,
        clearable=False,
        value='indv_wr_kde',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Plot Type:"),
    dcc.Dropdown(id="primary_kde_plot_type",
        options=kde_plot_type_options,
        multi=False,
        clearable=False,
        value='pdf',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_primary_kde_event",
            options=event_options,
            multi=True,
            clearable=False,
            value=['Downhill'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Gender:"),
    dcc.Dropdown(id="select_primary_kde_gender",
            options=gender_options,
            multi=True,
            clearable=False,
            value=['Men'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
],className='dash-selection-container flex-row')

######################################################################
# Here we create the violin plot chart
# filtering and downdown options
# ###################################################################
violin_selections_top = html.Div([
    html.Div([
    html.Label("View:"),
    dcc.Dropdown(id="violin_plot_type",
        options=violin_plot_type_options,
        multi=False,
        clearable=False,
        value='event_view',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_violin_event",
            options=event_options,
            multi=True,
            clearable=False,
            value=['Slalom','Downhill'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Gender:"),
    dcc.Dropdown(id="select_violin_gender",
            options=gender_options,
            multi=True,
            clearable=False,
            value=['Men'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
],className='dash-selection-container flex-row')

minimum_date_days = 365*5

######################################################################
# Here we create the primary kde plot chart
# age method selection options
# ###################################################################
primary_kde_age_options = html.Div([
    html.H3('Peak Age Method'),
    dcc.RadioItems(id='primary_kde_peak_age_method',options=peak_age_method_options, value='percentile',className='dash-radio-items'),
    dcc.RangeSlider(-3,3,1,value=[-1,1],marks=None,tooltip={"placement": "bottom"},id='primary_kde_peak_age_slider',className='kernel-slider')
],className='dash-selection-container flex-col',id="peak_age_container")

######################################################################
# Here we create the primary kde plot chart
# kernel options 1
# ###################################################################
primary_kde_kernel_options_1 = html.Div([
    html.H3(id='primary_kde_kernel_select_1_title'),
    html.Label('Kernel:'),
    dcc.Dropdown(id='primary_kde_kernel_select_1_dropdown',
    options=kernel_options,
    value='gaussian',
    clearable=False,
    multi=False),
    html.Label('Bandwidth:'),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="primary_kde_bandwidth_slider_1",
        className='kernel-slider')
    
],className='dash-selection-container flex-col')

######################################################################
# Here we create the primary kde plot chart
# kernel options 2
# ###################################################################
primary_kde_kernel_options_2 = html.Div([
    html.H3(id='primary_kde_kernel_select_2_title'),
    html.Label('Kernel:'),
    dcc.Dropdown(id='primary_kde_kernel_select_2_dropdown',
    options=kernel_options,
    value='gaussian',
    clearable=False,
    multi=False),
    html.Label('Bandwidth:'),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="primary_kde_bandwidth_slider_2",
        className='kernel-slider')
],className='dash-selection-container flex-col')

######################################################################
# Here we create the primary kde plot chart
# age method selection options
# ###################################################################
age_kde_type_selection = html.Div([html.Div([
    html.Label('Method: '),
    dcc.Dropdown(id='age_kde_type_select',
    options=age_range_options,
    value=1,
    clearable=False,
    multi=False),
],className='dash-container shelf w15 jc-center ai-center'
)],className='title-shelf')

######################################################################
# Here we create the primary kde comparison 1 plot 
# selection and method options
# ###################################################################
kde_comparison_selections_1 = html.Div([
    html.Div([
    html.Label("KDE Type:"),
    dcc.Dropdown(id="select_comparison_kde_method_1",
        options=kde_options,
        multi=False,
        clearable=False,
        value='indv_wr_kde',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_comparison_kde_event_1",
            options=event_options,
            multi=False,
            clearable=False,
            value=['Downhill'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Gender:"),
    dcc.Dropdown(id="select_comparison_kde_gender_1",
            options=gender_options,
            multi=False,
            clearable=False,
            value=['Men'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label('Kernel (Blue):'),
    dcc.Dropdown(id="select_comparison_kde_kernel_1_1",
        options=kernel_options,
        multi=False,
        clearable=False,
        value='gaussian',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label('Bandwidth (Blue)'),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="select_comparison_kde_bandwidth_1_1",
        className='kernel-slider')],className='selection-label-container flex-row'),
    html.Div([
    html.Label('Kernel (Red):'),
    dcc.Dropdown(id="select_comparison_kde_kernel_1_2",
        options=kernel_options,
        multi=False,
        clearable=False,
        value='gaussian',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Bandwidth (Red):"),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="select_comparison_kde_bandwidth_1_2",
        className='kernel-slider')],className='selection-label-container flex-row'),
],className='dash-selection-container flex-col')


######################################################################
# Here we create the primary kde comparison 2 plot 
# selection and method options
# ###################################################################
kde_comparison_selections_2 = html.Div([
    html.Div([
    html.Label("KDE Type:"),
    dcc.Dropdown(id="select_comparison_kde_method_2",
        options=kde_options,
        multi=False,
        clearable=False,
        value='indv_wr_kde',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Event:"),
    dcc.Dropdown(id="select_comparison_kde_event_2",
            options=event_options,
            multi=False,
            clearable=False,
            value=['Downhill'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Gender:"),
    dcc.Dropdown(id="select_comparison_kde_gender_2",
            options=gender_options,
            multi=False,
            clearable=False,
            value=['Men'],
            className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Kernel (Blue):"),
    dcc.Dropdown(id="select_comparison_kde_kernel_2_1",
        options=kernel_options,
        multi=False,
        clearable=False,
        value='gaussian',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Bandwidth (Blue):"),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="select_comparison_kde_bandwidth_2_1",
        className='kernel-slider')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Kernel (Red):"),
    dcc.Dropdown(id="select_comparison_kde_kernel_2_2",
        options=kernel_options,
        multi=False,
        clearable=False,
        value='gaussian',
        className='dash-dropdown')],className='selection-label-container flex-row'),
    html.Div([
    html.Label("Bandwidth (Red):"),
    dcc.Slider(0.5,1.5,0.01,value=1.0,marks=None,tooltip={"placement": "bottom"},
        id="select_comparison_kde_bandwidth_2_2",
        className='kernel-slider')],className='selection-label-container flex-row'),
],className='dash-selection-container flex-col')

######################################################################
# Here we create the full length title shelves for our 5
# primary sections
# ###################################################################
kde_age_results_title = TitleShelf(title='Optimal Peak Age Ranges Found By Kernel Density Estimation',title_sub_text='')
kde_cv_title_shelf = TitleShelf(title='5 Fold Cross-Validation of KDE Maximum Likelihood',title_sub_text='')
kde_violin_title_shelf = TitleShelf(title='Distribution of Medal Achievement by Event and Gender',title_sub_text='')
kde_title_shelf_1 = TitleShelf(title='Kernel Density Estimation of Peak Alpine Skiing Performance',title_sub_text='')
kde_comparison_title_shelf = TitleShelf(title='Kernel Density Estimation Comparison Tool',title_sub_text='')

######################################################################
# Here we create the callback targeted components which will all demonstrate
# loading functionality 
# ################################################################### 
#violin plot
primary_violin_figure = LoadingGraph(graph_id='primary_violin_plot',className='dash-container shelf w95')

#primary kde plot and age tables
primary_kde_figure = LoadingGraph(graph_id='primary_kde_plot',className='dash-container shelf w95')
kde_age_results_table_container = dcc.Loading(html.Div([],className='dash-container shelf w100 jc-center ai-start',id='kde_age_range_table_container'),parent_className='dash-container shelf w100 jc-center ai-start')

#kde cv plots
CV_kde_figure_1 = LoadingGraph(graph_id='cv_kde_plot_1',className='dash-container w35 mt50')
CV_kde_figure_2 = LoadingGraph(graph_id='cv_kde_plot_2',className='dash-container w35 mt50')
#kde comparison plots
kde_comparison_plot_1 = LoadingGraph(graph_id='kde_comparison_plot_1',className='dash-container w35 mt50')
kde_comparison_plot_2 = LoadingGraph(graph_id='kde_comparison_plot_2',className='dash-container w35 mt50')



###################################################################################### 
# House all of our pipeline ranking plots and components in a self contained shelf
# ################################################################################### 

#vioin plots
violin_container = html.Div([violin_selections_top,primary_violin_figure],className='dash-container stack w70 jc-center ai-center')
violin_shelf = html.Div([violin_container],className='dash-container w100 shelf jc-center')

#kde cv plots
cv_kde_container = html.Div([CV_kde_figure_1,CV_kde_figure_2],className='dash-container shelf w100 jc-center ai-center ac-center')

#primary kde plot and selection components
primary_kde_horizontal_container = html.Div([primary_kde_figure],className='dash-container shelf w100 jc-center',style={'min-height':'500px'})
primary_kde_selections_lower = html.Div(children=[primary_kde_kernel_options_1,primary_kde_age_options,primary_kde_kernel_options_2],className='dash-selection-container flex-row')
primary_kde_central_container = html.Div([primary_kde_selections_top,primary_kde_horizontal_container,primary_kde_selections_lower],className='dash-container stack w70 jc-center ai-center')
kde_primary_shelf = html.Div([primary_kde_central_container],className='dash-container w100 shelf jc-center')

#kde comparison components
kde_comparison_shelf = html.Div([kde_comparison_selections_1,kde_comparison_plot_1,kde_comparison_plot_2,kde_comparison_selections_2],className='dash-container shelf w100 jc-center ai-center ac-center')


######################################################################
# Here we take each of our section shelf elements and append them
# and post them to the application's layout in order of appearance
# ################################################################### 

layout = html.Div(children=[
    kde_violin_title_shelf,
    violin_shelf,
    kde_title_shelf_1,
    kde_primary_shelf,
    kde_cv_title_shelf,
    cv_kde_container,
    kde_age_results_title,
    age_kde_type_selection,
    kde_age_results_table_container,
    kde_comparison_title_shelf,
    kde_comparison_shelf

])

######################################################################
# Begin Application callback section
# ###################################################################

######################################################################
# callback updates and generates the kde age table based on user selections
# ################################################################### 
@callback(
    Output("kde_age_range_table_container","children"),
    Input('age_kde_type_select','value')
)
def Update_Age_Table_Ranges(mode):
    result = Generate_KDE_Age_Range_Results_Table(mode=mode)
    return result

######################################################################
# callback updates primary KDE plot based on selected filters
# ################################################################### 
@callback(
    [Output("select_primary_kde_event",'options'),
    Output("select_primary_kde_event",'value'),
    Output("select_primary_kde_gender",'options'),
    Output("select_primary_kde_gender",'value')],
    Input("select_primary_kde_method",'value')
)
def Update_Primary_KDE_Event_Gender_Dropdowns(kde_method):
    event_options = Get_Event_Options_From_KDE_Select(kde_method)
    default_event_value = event_options[0]['value']
    default_gender_value = gender_options[0]['value']
    return event_options,default_event_value,gender_options,default_gender_value

######################################################################
# callback updates primary KDE plot titles based on selections
# ################################################################### 
@callback(
    [Output("primary_kde_kernel_select_1_title","children"),
    Output("primary_kde_kernel_select_2_title","children")],
    [Input("select_primary_kde_method","value")]
)
def Update_Primary_KDE_Sub_Section_Titles(kde_method):
    if kde_method == 'event_kde':
        title_1_string = 'All Athletes KDE Parameter Selection'
        title_2_string = 'Top Athletes KDE Parameter Selection'
    elif kde_method == 'indv_wr_kde':
        title_1_string = 'Peak Individual World Record KDE Parameter Selection'
        title_2_string = 'Peak Individual Scaled World Record KDE Parameter Selection'
    elif kde_method == 'indv_rank_kde':
        title_1_string = 'Peak Individual Rank KDE Parameter Selection'
        title_2_string = 'Peak Individual Scaled Rank KDE Parameter Selection'
    else:
        title_1_string = 'Pre Regulation Top Athletes KDE Parameter Selection'
        title_2_string = 'Post Regulation Top Athletes KDE Parameter Selection'
    return title_1_string, title_2_string

######################################################################
# callback updates  KDE primary plots based on selected filters
# ################################################################### 
@callback(
    [Output("primary_kde_plot", "figure")],
    [Input("select_primary_kde_method",'value'),
    Input("select_primary_kde_gender",'value'),
    Input("select_primary_kde_event",'value'),
    Input('primary_kde_plot_type','value'),
    Input('primary_kde_kernel_select_1_dropdown','value'),
    Input('primary_kde_bandwidth_slider_1','value'),
    Input('primary_kde_kernel_select_2_dropdown','value'),
    Input('primary_kde_bandwidth_slider_2','value'),
    Input('primary_kde_peak_age_method','value'),
    Input('primary_kde_peak_age_slider','value')]
)
def Generate_Primary_KDE_Plot(kde_method,genders,events,plot_type,kernel_1,bw_1,kernel_2,bw_2,age_method,slider_value):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    kde_model_1 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=1,kernel=kernel_1,bandwidth=bw_1)
    kde_model_2 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=2,kernel=kernel_2,bandwidth=bw_2)
    fig = Generate_KDE_Chart_From_Models(kde_method=kde_method,plot_type=plot_type,age_method=age_method,age_range=slider_value,model_1=kde_model_1,model_2=kde_model_2)
    return [fig]

######################################################################
# callback updates KDE CV 1 plot based on selected filters
# ################################################################### 
@callback(
    Output('cv_kde_plot_1','figure'),
    [Input("select_primary_kde_method",'value'),
    Input("select_primary_kde_gender",'value'),
    Input("select_primary_kde_event",'value')]
)
def Generate_KDE_CV_Plot_1(kde_method,genders,events):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    cv_results = Generate_KDE_CV(df=df_copy,kde_method=kde_method,genders=genders,events=events,mode=1)
    cv_1 = cv_results
    cv_1_optimal_kernel,cv_1_optimal_bw = Parse_Optimal_Parameters_From_CV_Results(cv_1)
    fig1 = Generate_CV_Plot(cv_results=cv_1,cv_type=kde_method,events=events,genders=genders,kde_figure_index=0,optimal_kernel=cv_1_optimal_kernel,optimal_bw=cv_1_optimal_bw)
    return fig1


######################################################################
# callback updates KDE CV 2 plot based on selected filters
# ################################################################### 
@callback(
    Output('cv_kde_plot_2','figure'),
    [Input("select_primary_kde_method",'value'),
    Input("select_primary_kde_gender",'value'),
    Input("select_primary_kde_event",'value')]
)
def Generate_KDE_CV_Plot_2(kde_method,genders,events):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    cv_results = Generate_KDE_CV(df=df_copy,kde_method=kde_method,genders=genders,events=events,mode=2)
    cv_2 = cv_results
    cv_2_optimal_kernel,cv_2_optimal_bw = Parse_Optimal_Parameters_From_CV_Results(cv_2)
    fig2 = Generate_CV_Plot(cv_results=cv_2,cv_type=kde_method,events=events,genders=genders,kde_figure_index=1,optimal_kernel=cv_2_optimal_kernel,optimal_bw=cv_2_optimal_bw)
    return fig2


######################################################################
# callback shows/hides plot options based on selection
# ################################################################### 
@callback(
    Output("peak_age_container","className"),
    Input("primary_kde_plot_type","value")
)
def Show_Hide_Age_Select(chart_type):
    if chart_type == 'pdf':
        return 'dash-selection-container flex-col'
    else:
        return 'dash-selection-container flex-col hide'

######################################################################
# callback  updates age sliders based on kernel method
# ###################################################################
@callback(
    [Output('primary_kde_peak_age_slider','value'),
    Output('primary_kde_peak_age_slider','min'),
    Output('primary_kde_peak_age_slider','max'),
    Output('primary_kde_peak_age_slider','step')],
    Input("primary_kde_peak_age_method","value")
    
)
def Update_Age_Sider_Based_On_Age_Method_Selection(age_method):
    if age_method == 'std':
        min_val = -3
        max_val = 3
        step_val = 0.1
        start_vals = [-0.5,0.5]

    elif age_method == 'percentile':
        min_val = 0.0
        max_val = 1.0
        step_val = .05
        start_vals = [(.5-(1/6)),(.5+(1/6))]
    
    else:
        min_val = 15
        max_val = 45
        step_val = 1
        start_vals = [20,30]

    return start_vals, min_val, max_val, step_val

######################################################################
# callback generates and updates primary violin plot based on user selections
# ###################################################################
@callback(
    Output("primary_violin_plot","figure"),
    Input("violin_plot_type","value"),
    Input("select_violin_event","value"),
    Input("select_violin_gender","value")
)
def Generate_Violin_Plot(plot_type,events,genders):
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    if plot_type == 'event_view':
        fig = Generate_Event_Violin_Plot(df,events,genders)
    elif plot_type == 'indv_rank_view':
        fig = Generate_Indv_Rank_Violin_Plot(df,events,genders)
    elif plot_type == 'indv_wr_view':
        fig = Generate_Indv_World_Rank_Violin_Plot(df,events,genders)
    else:
        fig = Generate_Reg_Violin_Plot(df,events,genders)
    return fig

######################################################################
# callback generates kde comparison plot 1
# ###################################################################
@callback(Output(component_id='kde_comparison_plot_1',component_property='figure'),
[Input(component_id='select_comparison_kde_method_1',component_property='value'),
Input(component_id='select_comparison_kde_event_1',component_property='value'),
Input(component_id='select_comparison_kde_gender_1',component_property='value'),
Input(component_id='select_comparison_kde_kernel_1_1',component_property='value'),
Input(component_id='select_comparison_kde_bandwidth_1_1',component_property='value'),
Input(component_id='select_comparison_kde_kernel_1_2',component_property='value'),
Input(component_id='select_comparison_kde_bandwidth_1_2',component_property='value')]
)
def Update_KDE_Comparison_Plot_1(kde_method,events,genders,kern_1,bw_1,kern_2,bw_2):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    kde_model_1 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=1,kernel=kern_1,bandwidth=bw_1)
    kde_model_2 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=2,kernel=kern_2,bandwidth=bw_2)
    optimal_age_dict = Get_Event_Gender_Peak_Age_Bounds()
    optimal_age_lower = optimal_age_dict[genders[0]][events[0]]['Age Lower']
    optimal_age_upper = optimal_age_dict[genders[0]][events[0]]['Age Upper']
    age_range = [optimal_age_lower,optimal_age_upper]
    fig = Generate_KDE_Chart_From_Models(kde_method=kde_method,plot_type='pdf',age_method='percentile',age_range=age_range,model_1=kde_model_1,model_2=kde_model_2)
    return fig

######################################################################
# callback generates kde comparison plot 2
# ###################################################################
@callback(Output(component_id='kde_comparison_plot_2',component_property='figure'),
[Input(component_id='select_comparison_kde_method_2',component_property='value'),
Input(component_id='select_comparison_kde_event_2',component_property='value'),
Input(component_id='select_comparison_kde_gender_2',component_property='value'),
Input(component_id='select_comparison_kde_kernel_2_1',component_property='value'),
Input(component_id='select_comparison_kde_bandwidth_2_1',component_property='value'),
Input(component_id='select_comparison_kde_kernel_2_2',component_property='value'),
Input(component_id='select_comparison_kde_bandwidth_2_2',component_property='value')]
)
def Update_KDE_Comparison_Plot_2(kde_method,events,genders,kern_1,bw_1,kern_2,bw_2):
    df_copy = df.copy()
    if type(genders) != list:
        genders = [genders]
    if type(events) != list:
        events = [events]
    kde_model_1 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=1,kernel=kern_1,bandwidth=bw_1)
    kde_model_2 = Generate_KDE_Fit(df_copy,kde_method=kde_method,genders=genders,events=events,mode=2,kernel=kern_2,bandwidth=bw_2)
    optimal_age_dict = Get_Event_Gender_Peak_Age_Bounds()
    optimal_age_lower = optimal_age_dict[genders[0]][events[0]]['Age Lower']
    optimal_age_upper = optimal_age_dict[genders[0]][events[0]]['Age Upper']
    age_range = [optimal_age_lower,optimal_age_upper]
    fig = Generate_KDE_Chart_From_Models(kde_method=kde_method,plot_type='pdf',age_method='percentile',age_range=age_range,model_1=kde_model_1,model_2=kde_model_2)
    return fig

