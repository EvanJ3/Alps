import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pathlib
import plotly.express as px
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()

def Get_Dash_Country_Options(df):
    """Gets and formats potential noc's into dash dropdown format

    Args:
        df (dataframe): alpine event dataframe

    Returns:
        list of dicts: dash country dropdown options with labels as country name and values as USOPC Noc Abbv
    """
    df_copy = df.copy()
    df_copy = df_copy[['NOC Name','NOC']]
    option_array = np.array(df_copy.groupby(by=['NOC Name','NOC']).count().index)
    output_options = []
    for i in range(0,option_array.shape[0]):
        output_options.append({'label':option_array[i][0],'value':option_array[i][1]})
    return output_options

def Get_Dash_Athlete_Name_Options(df,selected_country,selected_gender):
    """Gets and returns dash dropdown options for athelets from a given selected
    gender and noc

    Args:
        selected_country(str):USOPC NOC abbv
        selected_gender(str): one of 'Men' 'Women'

    Returns:
        list of dicts:dash athlete dropdown options with labels as athlete name and values as USOPC person id
    """
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['NOC'] == selected_country)&(df_copy['Event Gender'] == selected_gender)]
    return [{'label':name, 'value':name} for name in sorted(df_copy.Competitor.unique())]

def Get_Specific_Group_Dataframe(df,event_name_short,gender='Men',min_rank=None,max_rank=None,class_name='Elite',filter_reg=True,reg_type='post'):
    """Gets and returns a filtered view of the alpine dataset according to various specified arguments

    Args:
        df (dataframe): alpine event dataframe
        events(str or list): list of event name short strings to filter on
        genders (str or list): genders to filter on
        min_rank (int): minimum rank to include (large rank is worse so min is the highest or worst rank included)
        max_rank(int): maximum rank to include (small rank is better so max is the lowest or best rank included)
        class_name (str): class designation of ['Elite','Junior" ,'Youth' or None]
        filter_reg(bool): filter on regulation change or return full dataset
        reg_type(str): ['pre' or 'post'] if filter_reg is True applies the corresponding filter

    Returns:
        dataframe: filtered alpine dataset view
    """
    df_copy = df.copy()
    df_copy = df_copy[df_copy['Event Name Short'] == event_name_short]
    df_copy = df_copy[df_copy['Event Gender'] == gender]
    df_copy = df_copy[df_copy['Class'] == class_name]
    if min_rank is not None:
        df_copy = df_copy[df_copy['Rank']<=min_rank]
    else:
        pass
    
    if max_rank is not None:
        df_copy = df_copy[df_copy['Rank']>=max_rank]
    else:
        pass
    
    if filter_reg:
        if reg_type == 'post':
            df_copy = df_copy[df_copy['Competition Date']>='2003-01-01']
        else:
            df_copy = df_copy[df_copy['Competition Date']<'2003-01-01']
    else:
        pass
    
    return df_copy


def Filter_For_Individual_Athlete_Events(df,country,gender,athlete_name,min_rank=10):
    """Gets and returns a dataframe containing only timed events for a given athlete and noc

    Args:
        df(dataframe): alpine events dataframe
        country(str): usopc noc country abbv of ahtlete
        genders(list or string): genders to filter on
        athlete_name (str): given athlete name to filter on 
        min_rank: minimum rank value to include

    Returns:
        dataframe: a dataframe of only timed events filtered on the given conditions
    """
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['NOC'] == country)&(df_copy['Event Gender'] == gender)&(df_copy['Competitor']==athlete_name)&(df_copy['Class']=='Elite')]
    df_copy = df_copy[['Class','Competition Date','Event Name Short','Rank','Athlete Age Days Derived','Season']]
    df_copy['Threshold_Rank'] = df_copy.apply(lambda x: min(x['Rank'],min_rank),axis=1)
    df_copy.sort_values(by='Competition Date',inplace=True)
    return df_copy


def Scrape_Athlete_Image(athlete_name,athlete_country,athlete_gender):
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
        src_out = None
    return src_out

def Get_Gender_Event_Timed_Events(df,gender,event='all',class_name='Elite',include_rankings=False):
    """Gets and returns a dataframe containing only timed events no world ranks or points events

    Args:
        df(dataframe): alpine events dataframe
        genders(list or string): genders to filter on
        events(list or string): list of event name short strings to filter on
        class_name(string): class designation ['Elite','Junior" ,'Youth' or None]
        include_rankings: where or not to include ranking events

    Returns:
        dataframe: a dataframe of only timed events filtered on the given conditions
    """
    df_copy = df.copy()
    if include_rankings:
        if event == 'all':
            df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['Class']==class_name)]
        else:
            df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['Event Name Short']==event)&(df_copy['Class']==class_name)]
            
    else:
        if event == 'all':
            df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')&(df_copy['Class']==class_name)]
        else:
            df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')&(df_copy['Event Name Short']==event)&(df_copy['Class']==class_name)]
    df_copy = df_copy[['Competition Date','Athlete Age Days Derived','Rank','Competitor','Person ID','Adj_Rank','Event Hash','Class','Event Name Short','Scaled_Rank']]
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    return df_copy

def Get_Athlete_Timed_Events(df,person_id,event='all'):
    """Gets and returns athlete timed events in a given discipline

    Args:
        df(dataframe): alpine events dataframe
        person_id(int): USOPC person ID
        event(str or list): events to filter on 

    Returns:
        dataframe: a dataframe of only timed events filtered on the given conditions
    """
    df_copy = df.copy()
    if event=='all':
        df_copy = df_copy[(df_copy['Person ID'] == person_id)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')&(df_copy['Class']=='Elite')]
    else:
        df_copy = df_copy[(df_copy['Person ID'] == person_id)&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')&(df_copy['Event Name Short']==event)&(df_copy['Class']=='Elite')]

    df_copy = df_copy[['Competition Date','Athlete Age Days Derived','Rank','Competitor','Person ID','Adj_Rank','Event Hash','Class','Event Name Short','Scaled_Rank']]
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    return df_copy


def max_min_scaling(x,x_min,x_max):
    """transform a given value to max-min standarization

    Args:
        x (float): value to max-min scale
        x_min(float): minimum scale value
        x_max(float): maximum scale value


    Returns:
        float: max min scaled value
    """
    return abs(1-((x-x_min)/(x_max-x_min)))

def Generate_Athlete_Rank_Diff_Scatterplot(df,event_df,person_id,person_name):
    """Generates and returns the indivudal athlete competition difficulty scatterplot

    Args:
        df(dataframe): alpine events dataframe
        person_id(int): USOPC person ID
        eventdf(dataframe): timed events dataframe
        person_name: athlete name

    Returns:
        dash figure: compeition rank scatterplot
    """
    event_lister = []
    diff_lister = []
    scale_lister = []
    indv_df = Get_Athlete_Timed_Events(df,person_id,event='all')
    indv_df = indv_df.merge(event_df,how='left',on='Event Hash')
    indv_df = indv_df.dropna()
    indv_df = indv_df[['Event Name Short','Scaled Difficulty','Scaled_Rank']]
    indv_df = indv_df[indv_df['Scaled_Rank']!=0.0]
    event_array = np.array(indv_df['Event Name Short']).flatten()
    diff_array = np.array(indv_df['Scaled Difficulty']).flatten()
    scale_array = np.array(indv_df['Scaled_Rank']).flatten()
    fig = go.Figure()
    fig = px.scatter(x=scale_array, y=diff_array, color=event_array,labels={'color':'Event'},range_x=[0.0,1.1],range_y=[0.0,1.1])
    chart_title_string = 'Career Competition Rank vs. Difficulty'
    fig.update_layout(title_text=chart_title_string)
    fig.update_layout(xaxis_title='Scaled Rank (last to first)',yaxis_title='Scaled Difficulty (low to high)')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    if event_array.shape[0] <= 0:
        no_entries_message = f"Not enough data for {person_name} Yet"
        fig.update_layout(annotations=[{"text": no_entries_message,"xref": "paper","yref": "paper","showarrow": False,"font": {"size": 16}}])

    return fig


def Get_Gender_Event_WR(df,gender,event,class_name='Elite',post_reg=None):
    """Gets and returns an filtered dataframe containing only the worldranking
    standings for a given event, gender, and class selection

    Args:
        df (dataframe): alpine data df
        genders (str or list): gender of format ['Men' 'Women']
        events (str or list): event name short to filter on
        class_name (str): class designation of ['Elite','Junior" ,'Youth' or None]
        post_reg (str): whether to filter post regulation events

    Returns:
        dataframe: world ranking entires dataframe
    """
    df_copy = df.copy()
    if post_reg == 'yes':
        df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['Event Name Short']==event)&(df_copy['Class']==class_name)&(df_copy['USOC Master Competition Set Name']=='Standing/Ranking List')&(df_copy['Season']>=2004)]
    elif post_reg =='no':
        df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['Event Name Short']==event)&(df_copy['Class']==class_name)&(df_copy['USOC Master Competition Set Name']=='Standing/Ranking List')&(df_copy['Season']<2004)]
    else:
        df_copy = df_copy[(df_copy['Event Gender'] == gender)&(df_copy['Event Name Short']==event)&(df_copy['Class']==class_name)&(df_copy['USOC Master Competition Set Name']=='Standing/Ranking List')]
    df_copy = df_copy[['Athlete Age Days Derived','Rank','Competitor','Person ID']]
    df_copy = df_copy[df_copy['Rank']!=-1]
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    return df_copy


def Filter_Timed_Event_Entries(df,min_entries=12):
    """Generates and returns filtered dataframe where only timed events with more than min_entries are incldued

    Args:
        df(dataframe): timed events dataframe
        min_entries(int): minimum number of competitiors a timed event must have to be included in result

    Returns:
        dataframe: returns filtered dataframe where only timed events with more than min_entries are incldued
    """
    df_copy = df.copy()
    df_grouped = df_copy.groupby(by=['Person ID'])['Person ID'].count()
    filtered_unique_ids = np.array(df_grouped[df_grouped.values>=min_entries].index)
    df_filtered = df_copy[df_copy['Person ID'].isin(filtered_unique_ids)]
    return df_filtered

def Custom_Sample_KDE(kde,n_samples,random_state=None):
    """Sample fit kde distribution on kernels not currently 
    supported by sklearn namely all other than gaussian

    Args:
        kde: fit kde sklearn model object
        n_samples: number of samples to generate
        random_state: sets random seed to use

    Returns:
        np.array: returns array of sampled kde observations
    """
    data = np.array(kde.tree_.data)
    rng = check_random_state(random_state)
    u = rng.uniform(0, 1, size=n_samples)
    if kde.tree_.sample_weight is None:
        i = (u * data.shape[0]).astype(np.int64)
    else:
        cumsum_weight = np.cumsum(np.asarray(kde.tree_.sample_weight))
        sum_weight = cumsum_weight[-1]
        i = np.searchsorted(cumsum_weight, u * sum_weight)

    if kde.kernel == "gaussian":
        return np.atleast_2d(rng.normal(data[i], kde.bandwidth))
    
    dim = data.shape[1]
    X = rng.normal(size=(n_samples, dim))
    lengths = row_norms(X, squared=False)

    if kde.kernel == "tophat":
        rads = rng.uniform(size=(n_samples)) ** (1 / dim)
    elif kde.kernel == "linear":
        rads = rng.beta(a=dim, b=2, size=(n_samples))
    elif kde.kernel == "epanechnikov":
        rads = np.sqrt(rng.beta(a=dim / 2, b=2, size=(n_samples)))
    elif kde.kernel == "exponential":
        rads = rng.gamma(dim, size=(n_samples))

    X = X * (kde.bandwidth * rads / lengths).reshape(-1, 1)
    return np.atleast_2d(data[i] + X)

def Generate_Bio_List_Component(label,value):
    """Generate an html dash list item from label and value

    Args:
        label (str): label for list item
        value(str): value to assign the label

    Returns:
        dash list element: returns html dash list element
    """
    label_span = html.Span(label,className='athlete-bio-list-label')
    value_span = html.Span(value,className='athlete-bio-list-value')
    list_element = html.Li([label_span,value_span])
    return list_element


def Percentile_Peak_Age_Range_Traces(percentile_range,samples,scores):
    """Given a percentile range, kde samples, and kde scored samples the 
    function generates a dash area fill trace to fill under the kde curve
    thus highlighting percentile range area to the left and right of a 
    distributinal mode

    Args:
        percentile_range (list): upper and lower percetile range bounds
        samples (np.array): kde samples as 1d np array
        scores (np.array): kde scored samples as 1d np array


    Returns:
        dash trace: area trace
    """
    samples = samples.squeeze()
    scores = scores.squeeze()
    samples_sorted_idxs = np.argsort(samples)
    samples_sorted = samples[samples_sorted_idxs]
    scores_sorted = scores[samples_sorted_idxs]
    peak_density_index = np.argmax(scores_sorted)
    percentile_range_lower = abs(0.5 - percentile_range[0])
    percentile_range_upper = abs(percentile_range[-1] - 0.5)

    scores_left_of_max = scores_sorted[:peak_density_index+1]
    scores_right_of_max = scores_sorted[peak_density_index:]
    

    if (percentile_range[0] > 0.5) and (percentile_range[-1] > 0.5):
        
    
        scores_right_summation = np.sum(scores_right_of_max)
        normalized_right_scores = (scores_right_of_max/scores_right_summation)/2
        cum_sum_scores_right = np.cumsum(normalized_right_scores)
        scores_right_lower_bound_index = scores_left_of_max.shape[0] + np.argmin(np.abs((cum_sum_scores_right - min(percentile_range_lower,percentile_range_upper))**2.0))
        scores_right_upper_bound_index = scores_left_of_max.shape[0] + np.argmin(np.abs((cum_sum_scores_right - max(percentile_range_lower,percentile_range_upper))**2.0))
        x_trace = samples_sorted[scores_right_lower_bound_index:scores_right_upper_bound_index+1]
        y_trace = scores_sorted[scores_right_lower_bound_index:scores_right_upper_bound_index+1]

    elif (percentile_range[0] < 0.5) and (percentile_range[-1] < 0.5):
        scores_left_of_max_inverse = scores_left_of_max[::-1]
        scores_left_summation = np.sum(scores_left_of_max_inverse)
        normalized_left_inverse_scores = (scores_left_of_max_inverse/scores_left_summation)/2
        cum_sum_scores_left_inverse = np.cumsum(normalized_left_inverse_scores)
        scores_left_inverse_lower_bound_index = scores_left_of_max.shape[0] - np.argmin(np.abs((cum_sum_scores_left_inverse - max(percentile_range_lower,percentile_range_upper))**2.0))
        scores_left_inverse_upper_bound_index = scores_left_of_max.shape[0] - np.argmin(np.abs((cum_sum_scores_left_inverse - min(percentile_range_lower,percentile_range_upper))**2.0))
        x_trace = samples_sorted[scores_left_inverse_lower_bound_index:scores_left_inverse_upper_bound_index+1]
        y_trace = scores_sorted[scores_left_inverse_lower_bound_index:scores_left_inverse_upper_bound_index+1]
    
    else:
        
        scores_summation_left = np.sum(scores_left_of_max)
        scores_summation_right = np.sum(scores_right_of_max)
        normalized_scores_left = (scores_left_of_max/scores_summation_left)/2
        normalized_scores_right = (scores_right_of_max/scores_summation_right)/2
        normalized_scores_left = normalized_scores_left[::-1]
        scores_left_cum_sum = np.cumsum(normalized_scores_left)
        scores_right_cum_sum = np.cumsum(normalized_scores_right)
        scores_left_lower_bound_index = scores_left_of_max.shape[0] - np.argmin(np.abs((scores_left_cum_sum - percentile_range_lower)**2.0))
        scores_right_lower_bound_index = scores_left_of_max.shape[0] + np.argmin(np.abs((scores_right_cum_sum - percentile_range_upper)**2.0))
        x_trace = samples_sorted[scores_left_lower_bound_index:scores_right_lower_bound_index+1]
        y_trace = scores_sorted[scores_left_lower_bound_index:scores_right_lower_bound_index+1]
        
    return x_trace,y_trace