import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pathlib
import plotly.express as px
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
import warnings
from dash import dcc, html
from apps.Dash_Utilities import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()



def Get_WR_DF(df,genders,events,class_name='Elite'):
    """Gets and returns an filtered dataframe containing only the worldranking
    standings for a given event, gender, and class selection

    Args:
        df (dataframe): alpine data df
        genders (str or list): gender of format ['Men' 'Women']
        events (str or list): event name short to filter on
        class_name (str): class designation of ['Elite','Junior" ,'Youth' or None]

    Returns:
        dataframe: world ranking entires dataframe
    """
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['Event Gender'].isin(genders))&(df_copy['Event Name Short'].isin(events))&(df_copy['Class']==class_name)&(df_copy['USOC Master Competition Set Name']=='Standing/Ranking List')]
    df_copy = df_copy[['Athlete Age Days Derived','Rank','Competitor','Person ID','Scaled_Rank','Event Name Short','Adj_Rank','Event Gender']]
    df_copy = df_copy[df_copy['Rank']!=-1]
    df_copy = df_copy.dropna()
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    return df_copy

def Get_Specific_Group_Dataframe(df,events,genders,min_rank=None,max_rank=None,class_name='Elite',filter_reg=True,reg_type='post'):
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
    df_copy = df_copy[df_copy['Event Name Short'].isin(events)]
    df_copy = df_copy[df_copy['Event Gender'].isin(genders)]
    df_copy = df_copy[df_copy['Class'] == class_name]
    if min_rank is not None:
        df_copy = df_copy[df_copy['Adj_Rank']<=min_rank]
    
    if max_rank is not None:
        df_copy = df_copy[df_copy['Adj_Rank']>=max_rank]

    if filter_reg:
        if reg_type == 'post':
            df_copy = df_copy[df_copy['Competition Date']>='2003-01-01']
        else:
            df_copy = df_copy[df_copy['Competition Date']<'2003-01-01']
    
    df_copy = df_copy[['Event Name Short','Event Gender','Class','Event Hash','Person ID','Adj_Rank','Scaled_Rank','Rank','Athlete Age Days Derived']]
    return df_copy

def Get_Timed_Events(df,genders,events,class_name='Elite',include_rankings=False):
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
        df_copy = df_copy[(df_copy['Event Gender'].isin(genders))&(df_copy['Event Name Short'].isin(events))&(df_copy['Class']==class_name)]
    else:
        df_copy = df_copy[(df_copy['Event Gender'].isin(genders))&(df_copy['USOC Master Competition Set Name']!='Standing/Ranking List')&(df_copy['Event Name Short'].isin(events))&(df_copy['Class']==class_name)]
    df_copy = df_copy[['Competition Date','Athlete Age Days Derived','Rank','Competitor','Person ID','Adj_Rank','Event Hash','Class','Event Name Short','Scaled_Rank','Event Gender']]
    df_copy = df_copy.sort_values(by='Athlete Age Days Derived')
    return df_copy

def Filter_Timed_Events(df,min_entries=12):
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
    
def Indv_World_Ranking_KDE_CV(df,genders,events,rank_type='Adj_Rank'):
    """runs and returns indv world ranking cv results

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type(string): whether to run on adj_rank or rank

    Returns:
        dataframe: KDE CV results frame
    """
    seed = 10
    np.random.seed(seed)
    kernels = ['gaussian', 'epanechnikov', 'exponential', 'linear']
    bandwidths = np.logspace(-1, .5, 25) 
    wr_df = Get_WR_DF(df=df,genders=genders,events=events,class_name='Elite')
    wr_grouped = wr_df.groupby(by=['Person ID','Event Name Short'])[rank_type].min().reset_index()
    person_id_array = np.array(wr_grouped['Person ID'])
    event_array = np.array(wr_grouped['Event Name Short'])
    best_rank_achieved = np.array(wr_grouped[rank_type])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_i = person_id_array[i]
        event_i = event_array[i]
        best_rank_i = best_rank_achieved[i]
        wr_df_copy = wr_df.copy()
        wr_df_copy = wr_df_copy[(wr_df_copy['Person ID'] == person_i)&(wr_df_copy[rank_type] == best_rank_i)&(wr_df_copy['Event Name Short'] == event_i)]
        age_list.append(wr_df_copy['Athlete Age Days Derived'].mean())
    age_array = np.array(age_list).reshape(-1,1)/365.25
    grid = GridSearchCV(KernelDensity(),param_grid={'kernel':kernels,'bandwidth': bandwidths},n_jobs=-1,cv=5,refit=False,return_train_score=False)
    grid.fit(age_array)
    cv_result_frame = pd.DataFrame(grid.cv_results_)
    cv_result_frame =cv_result_frame[['param_bandwidth','param_kernel','mean_test_score','std_test_score']]
    return cv_result_frame

def Fit_Indv_World_Ranking_KDE(df,genders,events,rank_type,kernel,bandwidth):
    """Fits the individual world ranking kde given hyper params

    Args:
         df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type(str): whether to use scaled or base rank
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value

    Returns:
        sklearn model object: returns fit kde model
    """
    seed = 10
    np.random.seed(seed)
    wr_df = Get_WR_DF(df=df,genders=genders,events=events,class_name='Elite')
    wr_grouped = wr_df.groupby(by=['Person ID','Event Name Short'])[rank_type].min().reset_index()
    person_id_array = np.array(wr_grouped['Person ID'])
    event_array = np.array(wr_grouped['Event Name Short'])
    best_rank_achieved = np.array(wr_grouped[rank_type])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_i = person_id_array[i]
        event_i = event_array[i]
        best_rank_i = best_rank_achieved[i]
        wr_df_copy = wr_df.copy()
        wr_df_copy = wr_df_copy[(wr_df_copy['Person ID'] == person_i)&(wr_df_copy[rank_type] == best_rank_i)&(wr_df_copy['Event Name Short'] == event_i)]
        age_list.append(wr_df_copy['Athlete Age Days Derived'].mean())
    age_array = np.array(age_list).reshape(-1,1)/365.25
    clf = KernelDensity(bandwidth=bandwidth,kernel=kernel)
    clf.fit(age_array)
    return clf


def Indv_Ranking_KDE_CV(df,genders,events,rank_type='Adj_Rank'):
    """Performs crossvalidation on indv rank kde

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type (str): whether to use rank or scaled rank

    Returns:
        dataframe: returns crossvalidation results as dataframe
    """
    seed = 10
    np.random.seed(seed)
    kernels = ['gaussian', 'epanechnikov', 'exponential', 'linear']
    bandwidths = np.logspace(-1, .5, 25)
    filtered_timed_events = Get_Timed_Events(df=df,genders=genders,events=events,class_name='Elite',include_rankings=False)
    filtered_timed_events = Filter_Timed_Events(filtered_timed_events,min_entries=6)
    filtered_timed_events = filtered_timed_events.dropna()
    filtered_timed_events = filtered_timed_events[['Person ID','Scaled_Rank','Athlete Age Days Derived','Adj_Rank','Event Name Short']]
    grouped_rank = filtered_timed_events.groupby(by=['Person ID','Event Name Short'])[rank_type].min().reset_index()
    rank_array = np.array(grouped_rank[rank_type])
    person_id_array = np.array(grouped_rank['Person ID'])
    event_array = np.array(grouped_rank['Event Name Short'])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_id_i = person_id_array[i]
        event_i = event_array[i]
        rank_i = rank_array[i]
        athlete_i_df = filtered_timed_events[(filtered_timed_events['Person ID']==person_id_i)&(filtered_timed_events[rank_type]==rank_i)&(filtered_timed_events['Event Name Short']==event_i)]
        age_list.append(athlete_i_df['Athlete Age Days Derived'].min())
    age_array = np.array(age_list).reshape(-1, 1)/365.25
    grid = GridSearchCV(KernelDensity(),param_grid={'kernel':kernels,'bandwidth': bandwidths},n_jobs=-1,cv=5,refit=False,return_train_score=False)
    grid.fit(age_array)
    cv_result_frame = pd.DataFrame(grid.cv_results_)
    cv_result_frame =cv_result_frame[['param_bandwidth','param_kernel','mean_test_score','std_test_score']]
    return cv_result_frame

def Fit_Indv_Ranking_KDE(df,genders,events,rank_type,kernel,bandwidth):
    """Fits and returns an indv ranking kde model according to selected params

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type(str): whether to use scaled or base rank
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value

    Returns:
        sklearn model: returns a fitted sklearn kde model for indv rank
    """
    seed = 10
    np.random.seed(seed)
    filtered_timed_events = Get_Timed_Events(df=df,genders=genders,events=events,class_name='Elite',include_rankings=False)
    filtered_timed_events = Filter_Timed_Events(filtered_timed_events,min_entries=6)
    filtered_timed_events = filtered_timed_events.dropna()
    filtered_timed_events = filtered_timed_events[['Person ID','Scaled_Rank','Athlete Age Days Derived','Adj_Rank','Event Name Short']]
    grouped_rank = filtered_timed_events.groupby(by=['Person ID','Event Name Short'])[rank_type].min().reset_index()
    rank_array = np.array(grouped_rank[rank_type])
    person_id_array = np.array(grouped_rank['Person ID'])
    event_array = np.array(grouped_rank['Event Name Short'])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_id_i = person_id_array[i]
        event_i = event_array[i]
        rank_i = rank_array[i]
        athlete_i_df = filtered_timed_events[(filtered_timed_events['Person ID']==person_id_i)&(filtered_timed_events[rank_type]==rank_i)&(filtered_timed_events['Event Name Short']==event_i)]
        age_list.append(athlete_i_df['Athlete Age Days Derived'].min())
    age_array = np.array(age_list).reshape(-1,1)/365.25
    clf = KernelDensity(bandwidth=bandwidth,kernel=kernel)
    clf.fit(age_array)
    return clf

def Regulation_KDE_CV(df,genders,events,reg_type='pre'):
    """Runs crossvlaidation on kde hyperparamters for regulation kde approach 

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         reg_type (str): whether to fit on pre or post reg data

    Returns:
        dataframe: return cross-validation results as a dataframe
    """
    seed = 10
    np.random.seed(seed)
    kernels = ['gaussian', 'epanechnikov', 'exponential', 'linear']
    bandwidths = np.logspace(-1, .5, 25)
    if reg_type == 'pre':
        reg_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=5,max_rank=1,class_name='Elite',filter_reg=True,reg_type='pre')
    else:
        reg_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=5,max_rank=1,class_name='Elite',filter_reg=True,reg_type='post')
    age_array = np.array(reg_df['Athlete Age Days Derived']).reshape(-1,1)/365.25
    grid = GridSearchCV(KernelDensity(),param_grid={'kernel':kernels,'bandwidth': bandwidths},n_jobs=-1,cv=5,refit=False,return_train_score=False)
    grid.fit(age_array)
    cv_result_frame = pd.DataFrame(grid.cv_results_)
    cv_result_frame =cv_result_frame[['param_bandwidth','param_kernel','mean_test_score','std_test_score']]
    return cv_result_frame

def Fit_Regulation_KDE(df,genders,events,reg_type,kernel,bandwidth):
    """Fit and returns a fit regualtion perspective kde model according to user params

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         reg_type(str): whether to use pre or post regulation fit
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value

    Returns:
        sklearn model: returns a fitted sklearn kde model
    """
    seed = 10
    np.random.seed(seed)
    if reg_type == 'pre':
        reg_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=5,max_rank=1,class_name='Elite',filter_reg=True,reg_type='pre')
    else:
        reg_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=5,max_rank=1,class_name='Elite',filter_reg=True,reg_type='post')
    age_array = np.array(reg_df['Athlete Age Days Derived']).reshape(-1,1)/365.25
    clf = KernelDensity(bandwidth=bandwidth,kernel=kernel)
    clf.fit(age_array)
    return clf

def Event_KDE_CV(df,genders,events,top_n=None):
    """Fit and returns a fit regualtion perspective kde model according to user params

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         top_n (int): number of postions to consider in the top portion of the kde default None

    Returns:
        dataframe: return cross-validation results as a dataframe
    """
    seed = 10
    np.random.seed(seed)
    kernels = ['gaussian', 'epanechnikov', 'exponential', 'linear']
    bandwidths = np.logspace(-1, .5, 25)
    if top_n is None:
        event_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=None,max_rank=4,class_name='Elite',filter_reg=False)
    else:
        event_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=top_n,max_rank=1,class_name='Elite',filter_reg=False)
    age_array = np.array(event_df['Athlete Age Days Derived']).reshape(-1,1)/365.25
    grid = GridSearchCV(KernelDensity(),param_grid={'kernel':kernels,'bandwidth': bandwidths},n_jobs=-1,cv=3,refit=False,return_train_score=False)
    grid.fit(age_array)
    cv_result_frame = pd.DataFrame(grid.cv_results_)
    cv_result_frame =cv_result_frame[['param_bandwidth','param_kernel','mean_test_score','std_test_score']]
    return cv_result_frame

def Fit_Event_KDE(df,genders,events,top_n,kernel,bandwidth):
    """Fit and returns a fit event perspective kde model according to user params

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         top_n(int): number of top positions to inlcude in top positon kde catagory
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value

    Returns:
        sklearn model: returns a fitted sklearn kde model
    """
    seed = 10
    np.random.seed(seed)
    kernels = ['gaussian', 'epanechnikov', 'exponential', 'linear']
    bandwidths = np.logspace(-1, .5, 25)
    if top_n is None:
        event_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=None,max_rank=4,class_name='Elite',filter_reg=False)
    else:
        event_df = Get_Specific_Group_Dataframe(df,events,genders,min_rank=top_n,max_rank=1,class_name='Elite',filter_reg=False)
    age_array = np.array(event_df['Athlete Age Days Derived']).reshape(-1,1)/365.25
    clf = KernelDensity(bandwidth=bandwidth,kernel=kernel)
    clf.fit(age_array)
    return clf


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
    np.random.seed(random_state)
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



def Generate_Event_KDE_CV(df,genders,events,mode):
    """Mid level entry point for running event level KDE CV
    handles mode determination

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         mode (bool) : whether to run full or top kde

    Returns:
        dataframe: cross-validation results in df
    """
    if mode == 1:
        output_cv = Event_KDE_CV(df,genders,events,top_n=3)
    else:
        output_cv = Event_KDE_CV(df,genders,events,top_n=None)
    return output_cv


def Generate_Indv_WR_KDE_CV(df,genders,events,mode):
    """Mid level entry point for running indv world ranking level KDE CV
    handles mode determination

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         mode (bool) : whether to run scaled or unscaled kde

    Returns:
        dataframe: cross-validation results in df
    """
    if mode == 1:
        output_cv = Indv_World_Ranking_KDE_CV(df=df,genders=genders,events=events,rank_type='Adj_Rank')
    else:
        output_cv = Indv_World_Ranking_KDE_CV(df=df,genders=genders,events=events,rank_type='Scaled_Rank')
    return output_cv

def Generate_Indv_Rank_KDE_CV(df,genders,events,mode):
    """Mid level entry point for running indv rank level KDE CV
    handles mode determination

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         mode (bool) : whether to run scaled or unscaled kde

    Returns:
        dataframe: cross-validation results in df
    """
    if mode == 1:
        output_cv = Indv_Ranking_KDE_CV(df,genders,events,rank_type='Adj_Rank')
    else:
        output_cv = Indv_Ranking_KDE_CV(df,genders,events,rank_type='Scaled_Rank')
    return output_cv


def Generate_Regulation_KDE_CV(df,genders,events,mode):
    """Mid level entry point for running  regulation level KDE CV
    handles mode determination

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         mode (bool) : whether to run pre or post regulation kde

    Returns:
        dataframe: cross-validation results in df
    """
    if mode == 1:
        output_cv = Regulation_KDE_CV(df,genders,events,reg_type='pre')
    else:
        output_cv = Regulation_KDE_CV(df,genders,events,reg_type='post')
    return output_cv


def Generate_KDE_CV(df,kde_method,genders,events,mode):
    """Top level entry point function for running of all kde cases

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         kde_method(str): one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run
         mode(int): which mode to run cv in from the given mid level function used

    Returns:
        dataframe: cross-validation results in df
    """
    if kde_method == 'event_kde':
        cv_results = Generate_Event_KDE_CV(df,genders,events,mode)
    elif kde_method == 'indv_wr_kde':
        cv_results = Generate_Indv_WR_KDE_CV(df,genders,events,mode)
    elif kde_method == 'indv_rank_kde':
        cv_results = Generate_Indv_Rank_KDE_CV(df,genders,events,mode)
    else:
        cv_results = Generate_Regulation_KDE_CV(df,genders,events,mode)
    return cv_results

def Get_Event_Options_From_KDE_Select(kde_method):
    """Given a kde method chosen returns the potential event options to populate the correspoinding dropdown box with

    Args:
        kde_method (str):one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run

    Returns:
        list of dicts: dash dropdown options
    """
    event_options = [
    {'label':'Downhill','value':'Downhill'},
    {'label':'Giant Slalom','value':'Giant Slalom'},
    {'label':'Slalom','value':'Slalom'},
    {'label':'Super G','value':'Super G'}]

    if kde_method == 'indv_wr_kde':
        event_options.append({'label':'Overall','value':'Overall'}) 
    else:
        event_options.append({'label':'Combination','value':'Combination'})

    return event_options

def Parse_Optimal_Parameters_From_CV_Results(cv_results):
    """Parses the contents of a KDE CV results dataframe and returns the optional 
    bandwidth and kernel hyperparameters of interest

    Args:
        cv_results(dataframe): KDE cross-validation results dataframe

    Returns:
        str: optimal kde kernel
        float: optimal kernel bandwidth
    """
    cv_df_copy = cv_results.copy()
    best_test_score = cv_df_copy['mean_test_score'].max()
    optimal_bandwidth = cv_df_copy[cv_df_copy['mean_test_score'] == best_test_score]['param_bandwidth'].iloc()[0]
    optimal_kernel = cv_df_copy[cv_df_copy['mean_test_score'] == best_test_score]['param_kernel'].iloc()[0]
    return optimal_kernel,optimal_bandwidth



def Generate_CV_Plot_Title(kde_method,events,genders):
    """Returns the proper plot title for a given set of parameters

    Args:
         genders(list or string): genders to filter on
         kde_method (str):one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run
         events(list or string): list of event name short strings to filter on
    Returns:
        str: returns plot's corresponding title
    """
    if kde_method == 'event_kde':
        kde_method_substring_1 = 'Event Level All Athletes Cross-Validation'
        kde_method_substring_2 = 'Event Level Top Athletes Cross-Validation'
    elif kde_method == 'indv_wr_kde':
        kde_method_substring_1 = 'Individual Peak Rank Cross-Validation'
        kde_method_substring_2 = 'Individual Peak Scaled Rank Cross-Validation'
    elif kde_method == 'indv_rank_kde':
        kde_method_substring_1 = 'Individual Peak World Ranking Cross-Validation'
        kde_method_substring_2 = 'Individual Peak Scaled World Ranking Cross-Validation'
    else:
        kde_method_substring_1 = 'Pre Regulation Top Athletes Cross-Validation'
        kde_method_substring_2 = 'Post Regulation Top Athletes Cross-Validation'

    if len(genders) == 2:
        gender_sub_string = 'All'
    else:
        gender_sub_string = genders[0]+"'s"

    if len(events) == 1:
        events_sub_string = events[0]
    elif len(events) == 2:
        events_sub_string = events[0] + ' & ' + events[1]
    else:
        events_sub_string = ''
        max_event_index = len(events)
        for i in range(0,len(events)):
            event_sub_string_i = events[i]
            if i == max_event_index-1:
                event_sub_string_i = '& ' + event_sub_string_i
            elif i== max_event_index:
                event_sub_string_i = event_sub_string_i 
            else:
                event_sub_string_i = event_sub_string_i + ', '
            events_sub_string += event_sub_string_i
    output_string_1 = gender_sub_string + ' ' + events_sub_string + ' ' + kde_method_substring_1
    output_string_2 = gender_sub_string + ' ' + events_sub_string + ' ' + kde_method_substring_2
    return output_string_1,output_string_2

def Generate_CV_Plot(cv_results,cv_type,events,genders,kde_figure_index,optimal_kernel,optimal_bw):
    """Generates KDE CV Plot

    Args:
        cv_results(dataframe):crossvalidation results as df
        events(list str): events to include
        genders(list str): genders to include
        kde_figure_index(int): which cv figure to plot against determing test vs. train
        optimal_kernel(str): identified optimal kernel type
        optimal_bw(float): identified optimal kernel bandwidth

    Returns:
        Dash Figure: returns KDE CV plot
    """
    cv_df = cv_results.copy()
    fig = go.Figure()
    title_strings = Generate_CV_Plot_Title(cv_type,events,genders)
    title_string = title_strings[kde_figure_index]
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear']
    for i in range(0,len(kernels)):
        kernel_i = kernels[i]
        cv_df_filtered = cv_df[cv_df['param_kernel']==kernel_i]
        cv_df_filtered = cv_df_filtered.sort_values('param_bandwidth')
        fig.add_trace(go.Scatter(x=cv_df_filtered['param_bandwidth'],y=cv_df_filtered['mean_test_score'],name=kernel_i))
    optimal_mle = cv_df[(cv_df['param_kernel']==optimal_kernel)&(cv_df['param_bandwidth']==optimal_bw)]['mean_test_score'].iloc()[0]
    optimal_text = 'Optimal Kernel: '+ optimal_kernel + '<br>'+ 'Optimal Bandwidth: ' + str(round(optimal_bw,2))
    fig.add_trace(go.Scatter(x=[optimal_bw],y=[optimal_mle],mode='markers+text',name='Maximum Likelihood',text=[optimal_text],textposition="bottom right"))
    fig.update_layout(title=title_string,xaxis_title='Bandwidth',yaxis_title='Maximum Liklihood',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return fig

def Generate_Event_KDE_Fit(df,genders,events,mode,kernel,bandwidth):
    """Midlevel function entry for determineing and running the required version of the event kde model

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on

    Returns:
        sklearn model object: returns the fitted sklearn event perspective model
    """
    if mode == 1:
        fit_kde_clf = Fit_Event_KDE(df,genders,events,top_n=3,kernel=kernel,bandwidth=bandwidth)
    else:
        fit_kde_clf = Fit_Event_KDE(df,genders,events,top_n=None,kernel=kernel,bandwidth=bandwidth)
    return fit_kde_clf


def Generate_Indv_WR_KDE_Fit(df,genders,events,mode,kernel,bandwidth):
    """Midlevel function entry for determineing and running the required version of the indv wr kde model

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value
         mode(int): whether to run in scaled or unscaled mode

    Returns:
        list: a list of strings representing the header columns
    """
    if mode == 1:
        fit_kde_clf = Fit_Indv_World_Ranking_KDE(df=df,genders=genders,events=events,rank_type='Adj_Rank',kernel=kernel,bandwidth=bandwidth)
    else:
        fit_kde_clf = Fit_Indv_World_Ranking_KDE(df=df,genders=genders,events=events,rank_type='Scaled_Rank',kernel=kernel,bandwidth=bandwidth)
    return fit_kde_clf

def Generate_Indv_Rank_KDE_Fit(df,genders,events,mode,kernel,bandwidth):
    """Midlevel function entry for determineing and running the required version of the indv rank kde model

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value
         mode(int): which kde run mode to use

    Returns:
        sklearn model object: returns the fitted sklearn event perspective model
    """
    if mode == 1:
        fit_kde_clf = Fit_Indv_Ranking_KDE(df,genders,events,rank_type='Adj_Rank',kernel=kernel,bandwidth=bandwidth)
    else:
        fit_kde_clf = Fit_Indv_Ranking_KDE(df,genders,events,rank_type='Scaled_Rank',kernel=kernel,bandwidth=bandwidth)
    return fit_kde_clf


def Generate_Regulation_KDE_Fit(df,genders,events,mode,kernel,bandwidth):
    """Midlevel function entry for determineing and running the required version of the event kde model

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value
         mode(int): which kde run mode to use
    Returns:
        sklearn model object: returns the fitted sklearn event perspective model
    """
    if mode == 1:
        fit_kde_clf = Fit_Regulation_KDE(df,genders,events,reg_type='pre',kernel=kernel,bandwidth=bandwidth)
    else:
        fit_kde_clf = Fit_Regulation_KDE(df,genders,events,reg_type='post',kernel=kernel,bandwidth=bandwidth)
    return fit_kde_clf

def Generate_KDE_Fit(df,kde_method,genders,events,mode,kernel,bandwidth):
    """Top level function for generating the required specificed kde fit

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         kernel (str): desired kde kernel see sklearn for options
         bandwidth(float): desired kde bandwidth value
         mode(int): which kde run mode to use

    Returns:
        sklearn model object: returns the fitted sklearn event perspective model
    """
    if kde_method == 'event_kde':
        fit_kde_clf = Generate_Event_KDE_Fit(df=df,genders=genders,events=events,mode=mode,kernel=kernel,bandwidth=bandwidth)
    elif kde_method == 'indv_wr_kde':
        fit_kde_clf = Generate_Indv_WR_KDE_Fit(df=df,genders=genders,events=events,mode=mode,kernel=kernel,bandwidth=bandwidth)
    elif kde_method == 'indv_rank_kde':
        fit_kde_clf = Generate_Indv_Rank_KDE_Fit(df=df,genders=genders,events=events,mode=mode,kernel=kernel,bandwidth=bandwidth)
    else:
        fit_kde_clf = Generate_Regulation_KDE_Fit(df=df,genders=genders,events=events,mode=mode,kernel=kernel,bandwidth=bandwidth)
    return fit_kde_clf

def Generate_Range_Traces(method,bounds,samples,scores):
    """returns the subset of x and y to include in the area plotting of peak age

    Args:
        samples(np.array): kde sampeles to operate over
        scores(np.array): kde scores to operate over
        method(str): range mode one of std percentile or discrete

    Returns:
        np.array: x subset of obersvations used to integrate over
        np.array: y subset of obersvations used to integrate over
    """
    samples = samples.squeeze()
    scores = scores.squeeze()
    if method == 'std':
        x_trace,y_trace = STD_Peak_Age_Range_Traces(sigma_range=bounds,samples=samples,scores=scores)
        
    elif method == 'percentile':
        x_trace,y_trace = Percentile_Peak_Age_Range_Traces(percentile_range=bounds,samples=samples,scores=scores)
    else:
        x_trace,y_trace = Discrete_Peak_Age_Range_Traces(discrete_range=bounds,samples=samples,scores=scores)
    return x_trace,y_trace

def Generate_Primary_KDE_Figure_Title(kde_method,plot_type):
    """Generates corresponding kde figure title

    Args:
        kde_method (str):one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run
        plot_type (str): determines type of plot to return one of ['cdf', 'pdf', 'deltapdf']

    Returns:
        str: kde chart title string
    """
    kde_type_string_dict ={'event_kde':'Event Level KDE',
                            'indv_wr_kde':'Peak World Rank KDE',
                            'indv_rank_kde':'Personal Best Rank',
                            'reg_kde':'Impact of Regulation KDE'}

    plot_type_string_dict = {'cdf':'Cumulative Density Plot',
                            'pdf':'Probablity Density Plot',
                            'delta':'Differnce of Probability Density Plot'}
    prefix = kde_type_string_dict[kde_method]
    suffix = plot_type_string_dict[plot_type]
    output_string = prefix + ' ' + suffix
    return output_string

def STD_Peak_Age_Range_Traces(sigma_range,samples,scores):
    """returns the subset of x and y to include in the area plotting of peak age

    Args:
         samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        np.array: x subset of obersvations used to integrate over
        np.array: y subset of obersvations used to integrate over
    """
    samples = samples.squeeze()
    scores = scores.squeeze()
    samples_sorted_idxs = np.argsort(samples)
    samples_sorted = samples[samples_sorted_idxs]
    scores_sorted = scores[samples_sorted_idxs]

    max_density_index = np.argmax(scores_sorted)
    sample_max_density = samples_sorted[max_density_index]
    score_max_density = scores_sorted[max_density_index]
    sample_std = np.std(samples)

    sigma_range_lower = np.abs(sigma_range[0])
    sigma_range_upper = np.abs(sigma_range[-1])

    samples_lower_bound = sample_max_density - (sigma_range_lower*sample_std)
    samples_upper_bound = sample_max_density + (sigma_range_upper*sample_std)

    trace_lower_bound_index = np.argmin(np.abs((samples_sorted - samples_lower_bound)**2.0))
    trace_upper_bound_index = np.argmin(np.abs((samples_sorted - samples_upper_bound)**2.0))

    x_trace = samples_sorted[trace_lower_bound_index:trace_upper_bound_index+1]
    y_trace = scores_sorted[trace_lower_bound_index:trace_upper_bound_index+1]
    return x_trace,y_trace


def Percentile_Peak_Age_Range_Traces(percentile_range,samples,scores):
    """returns the subset of x and y to include in the area plotting of peak age

    Args:
        samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        np.array: x subset of obersvations used to integrate over
        np.array: y subset of obersvations used to integrate over
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

def Discrete_Peak_Age_Range_Traces(discrete_range,samples,scores):
    """returns the subset of x and y to include in the area plotting of peak age

    Args:
        discrete_range(list of floats): user supplied age bounds
        samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        np.array: x subset of obersvations used to integrate over
        np.array: y subset of obersvations used to integrate over
    """
    samples = samples.squeeze()
    scores = scores.squeeze()

    samples_sorted_idxs = np.argsort(samples)
    samples_sorted = samples[samples_sorted_idxs]
    scores_sorted = scores[samples_sorted_idxs]

    samples_lower_bound = discrete_range[0]
    samples_upper_bound = discrete_range[-1]

    samples_lower_trace_index = np.argmin(np.abs((samples_sorted - samples_lower_bound)**2.0))
    samples_upper_trace_index = np.argmin(np.abs((samples_sorted - samples_upper_bound)**2.0))

    x_trace = samples_sorted[samples_lower_trace_index:samples_upper_trace_index+1]
    y_trace = scores_sorted[samples_lower_trace_index:samples_upper_trace_index+1]
    return x_trace,y_trace

def Generate_KDE_PDF_Traces(samples,scores):
    """Generates the ordered samples and socred samples into a pdf trace

    Args:
        samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        dash trace: returns the corresponding kde pdf trace
    """
    samples = samples.squeeze()
    scores = scores.squeeze()
    samples_sorted_idxs = np.argsort(samples)
    samples_sorted = samples[samples_sorted_idxs]
    scores_sorted = scores[samples_sorted_idxs]
    x_trace = samples_sorted
    y_trace = scores_sorted
    return x_trace,y_trace

def Generate_KDE_CDF_Traces(samples,scores):
    """Generates the ordered samples and socred samples into a cdf trace

    Args:
        samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        dash trace: returns the corresponding kde cdf trace
    """
    samples = samples.squeeze()
    scores = scores.squeeze()
    samples_sorted_idxs = np.argsort(samples)
    samples_sorted = samples[samples_sorted_idxs]
    scores_sorted = scores[samples_sorted_idxs]

    density_total_sum = np.sum(scores_sorted)
    normalized_density_scores = scores_sorted/density_total_sum
    y_trace = np.cumsum(normalized_density_scores)
    x_trace = samples_sorted
    return x_trace,y_trace

def Generate_KDE_Density_Trace_Label(kde_method,mode):
    """Returns a lengend labels for a given denisty plot

    Args:
        kde_method (str):one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run
        mode(str): which kdemethod subset mode to run in

    Returns:
        str: label string for legend entry of given density trace
    """
    if kde_method == 'event_kde':
        if mode == 1:
            trace_label = 'Top Athletes'
        else:
            trace_label = 'All Athletes'
    elif kde_method == 'indv_wr_kde':
        if mode == 1:
            trace_label = 'Peak Rank'
        else:
            trace_label = 'Peak Scaled Rank'
    elif kde_method == 'indv_rank_kde':
        if mode == 1:
            trace_label = 'Peak Rank'
        else:
            trace_label = 'Peak Scaled Rank'
    else:
        if mode == 1:
            trace_label = 'Pre-Reg Top Athletes'
        else:
            trace_label = 'Post-Reg Top Athletes'
    return trace_label

def Generate_Padded_Traces(trace_1_x,trace_1_y,trace_2_x,trace_2_y):
    """Pads and returns zero adjusted and padded traces to ensure both 
    traces plot over the same probablity range

    Args:
        trace_1_x (np.array): plot x trace 1
        trace_1_y (np.array): plot y trace 1
        trace_2_x (np.array): plot x trace 2
        trace_2_y (np.array): plot y trace 2

    Returns:
        np.array: x1 zero padded of obersvations
        np.array: y1 zero padded set of obersvations
        np.array: x2 zero padded of obersvations
        np.array: y2 zero padded set of obersvations
    """
    max_x_1 = np.max(trace_1_x)
    min_x_1 = np.min(trace_1_x)

    max_x_2 = np.max(trace_2_x)
    min_x_2 = np.min(trace_2_x)

    chart_min_idx_x = np.argmin(np.array([min_x_1,min_x_2]))
    chart_max_idx_x = np.argmax(np.array([max_x_1,max_x_2]))

    
    if chart_min_idx_x == 0:
        padd_x_range = trace_1_x[0:np.argmin(np.abs((trace_1_x - trace_2_x[0]))**2.0)]
        padd_y_range = np.zeros(padd_x_range.shape)
        trace_2_x = np.concatenate((padd_x_range,trace_2_x))
        trace_2_y = np.concatenate((padd_y_range,trace_2_y))

    else:
        padd_x_range = trace_2_x[0:np.argmin(np.abs((trace_2_x - trace_1_x[0]))**2.0)]
        padd_y_range = np.zeros(padd_x_range.shape)
        trace_1_x = np.concatenate((padd_x_range,trace_1_x))
        trace_1_y = np.concatenate((padd_y_range,trace_1_y))

    if chart_max_idx_x == 0:
        padd_x_range = trace_1_x[np.argmin(np.abs((trace_1_x - trace_2_x[-1]))**2.0):]
        padd_y_range = np.zeros(padd_x_range.shape)
        trace_2_x = np.concatenate((trace_2_x,padd_x_range))
        trace_2_y = np.concatenate((trace_2_y,padd_y_range))

    else:
        padd_x_range = trace_2_x[np.argmin(np.abs((trace_2_x - trace_1_x[-1]))**2.0):]
        padd_y_range = np.zeros(padd_x_range.shape)
        trace_1_x = np.concatenate((trace_1_x,padd_x_range))
        trace_1_y = np.concatenate((trace_1_y,padd_y_range))

    return trace_1_x,trace_1_y,trace_2_x,trace_2_y


def Generate_KDE_Chart_From_Models(kde_method,plot_type,age_method,age_range,model_1,model_2):
    """Gets and prints the spreadsheet's header columns

    Args:
        kde_method (str):one of [event_kde indv_rank_kde indv_wr_rank_kde or reg_kde] determines which cv perspecitve to run
        plot_type(str): ope of plot to generate one of 'cdf' 'pdf' 'deltapdf'
        age_method(str): determines age method to apply 'percentile' 'std' or 'discrete'
        age_range(list of float): upper and lower age bounds
        model_1(sklearn kde model object): first kde model used in the plot
        model_2(sklearn kde model object): second kde model used in the plot

    Returns:
        Dash Figure: returns dash figure for primary kde plot
    """
    seed = 10
    np.random.seed(seed)
    fig = go.Figure()
    title_string = Generate_Primary_KDE_Figure_Title(kde_method=kde_method,plot_type=plot_type)
    line_opacity = '1'
    area_opacity = '.3'
    model_1_line_color = f"rgba(92,138,228,{line_opacity})"
    model_2_line_color = f"rgba(245,32,32,{line_opacity})"
    model_1_area_color = f"rgba(92,138,228,{area_opacity})"
    model_2_area_color = f"rgba(245,32,32,{area_opacity})"
    model_1_samples = Custom_Sample_KDE(model_1,n_samples=10000,random_state=seed)
    model_1_scored_samples = np.exp(model_1.score_samples(model_1_samples))
    model_2_samples = Custom_Sample_KDE(model_2,n_samples=10000,random_state=seed)
    model_2_scored_samples = np.exp(model_2.score_samples(model_2_samples))
    
    if plot_type == 'pdf':
        x_axis_title_string = 'Age in Years'
        y_axis_title_string = 'Probablity Density'

        model_1_x_trace,model_1_y_trace = Generate_KDE_PDF_Traces(model_1_samples,model_1_scored_samples)
        model_1_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=1)+ ' Density'
        
        model_2_x_trace,model_2_y_trace = Generate_KDE_PDF_Traces(model_2_samples,model_2_scored_samples)
        model_2_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=2)

        model_1_x_trace_padded,model_1_y_trace_padded,model_2_x_trace_padded,model_2_y_trace_padded = Generate_Padded_Traces(model_1_x_trace,model_1_y_trace,model_2_x_trace,model_2_y_trace)
        fig.add_trace(go.Scatter(x=model_1_x_trace_padded,y=model_1_y_trace_padded,name=model_1_trace_label,mode="lines",marker=dict(line=dict(color=model_1_line_color,width=2))))
        fig.add_trace(go.Scatter(x=model_2_x_trace_padded,y=model_2_y_trace_padded,name=model_2_trace_label,mode="lines",marker=dict(line=dict(color=model_2_line_color,width=2))))

        
        model_1_area_x_trace,model_1_area_y_trace = Generate_Range_Traces(method=age_method,bounds=age_range,samples=model_1_samples,scores=model_1_scored_samples)
        model_1_area_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=1) + ' Peak Age'
        
        fig.add_trace(go.Scatter(x=model_1_area_x_trace,y=model_1_area_y_trace,fillcolor=model_1_area_color,fill='tozeroy',line_color=model_1_area_color,name=model_1_area_trace_label,mode="none",marker=dict(line=dict(color=model_1_area_color,width=2))))


        model_2_area_x_trace,model_2_area_y_trace = Generate_Range_Traces(method=age_method,bounds=age_range,samples=model_2_samples,scores=model_2_scored_samples)
        model_2_area_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=2) + ' Peak Age'
        fig.add_trace(go.Scatter(x=model_2_area_x_trace,y=model_2_area_y_trace,fill='tozeroy',fillcolor=model_2_area_color,name=model_2_area_trace_label,mode="none",marker=dict(line=dict(color=model_2_area_color,width=2)),visible="legendonly"))


    elif plot_type == 'cdf':
        x_axis_title_string = 'Age in Years'
        y_axis_title_string = 'Cumulative Density'
        model_1_sample_min = np.min(model_1_samples)
        model_1_sample_max = np.max(model_1_samples)
        model_2_sample_min = np.min(model_2_samples)
        model_2_sample_max = np.max(model_2_samples)
        age_min = min(model_1_sample_min,model_2_sample_min)
        age_max = max(model_1_sample_max,model_2_sample_max)
        model_1_aligned_samples = np.linspace(age_min,age_max,1000)[:,np.newaxis]
        model_2_aligned_samples = np.linspace(age_min,age_max,1000)[:,np.newaxis]
        model_1_aligned_scored_samples = np.exp(model_1.score_samples(model_1_aligned_samples))
        model_2_aligned_scored_samples = np.exp(model_2.score_samples(model_2_aligned_samples))
        model_1_x_trace,model_1_y_trace = Generate_KDE_CDF_Traces(model_1_aligned_samples,model_1_aligned_scored_samples)
        model_2_x_trace,model_2_y_trace = Generate_KDE_CDF_Traces(model_2_aligned_samples,model_2_aligned_scored_samples)
        model_2_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=2)
        model_1_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=1) + ' Density'
        fig.add_trace(go.Scatter(x=model_1_x_trace,y=model_1_y_trace,name=model_1_trace_label,mode="lines",marker=dict(line=dict(color=model_1_line_color,width=2))))
        fig.add_trace(go.Scatter(x=model_2_x_trace,y=model_2_y_trace,name=model_2_trace_label,mode="lines",marker=dict(line=dict(color=model_2_line_color,width=2))))

    else:
        x_axis_title_string = 'Age in Years'
        y_axis_title_string = 'Probablity Density Difference'
        model_1_sample_min = np.min(model_1_samples)
        model_1_sample_max = np.max(model_1_samples)
        model_2_sample_min = np.min(model_2_samples)
        model_2_sample_max = np.max(model_2_samples)
        age_min = min(model_1_sample_min,model_2_sample_min)
        age_max = max(model_1_sample_max,model_2_sample_max)
        model_1_aligned_samples = np.linspace(age_min,age_max,1000)[:,np.newaxis]
        model_2_aligned_samples = np.linspace(age_min,age_max,1000)[:,np.newaxis]
        model_1_aligned_scored_samples = np.exp(model_1.score_samples(model_1_aligned_samples))
        model_2_aligned_scored_samples = np.exp(model_2.score_samples(model_2_aligned_samples))
        model_1_x_trace,model_1_y_trace = Generate_KDE_PDF_Traces(model_1_aligned_samples,model_1_aligned_scored_samples)
        model_2_x_trace,model_2_y_trace = Generate_KDE_PDF_Traces(model_2_aligned_samples,model_2_aligned_scored_samples)
        difference_trace = model_1_y_trace - model_2_y_trace
        positive_y_trace = np.where(difference_trace<0,difference_trace,None)
        negative_y_trace = np.where(difference_trace>0,difference_trace,None)
        model_1_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=1)+ ' Density'
        model_2_trace_label = Generate_KDE_Density_Trace_Label(kde_method=kde_method,mode=2)+ ' Density'
        fig.add_trace(go.Scatter(x=model_1_x_trace,y=positive_y_trace,name=model_1_trace_label,mode="lines",marker=dict(line=dict(color=model_1_line_color,width=2))))
        fig.add_trace(go.Scatter(x=model_1_x_trace,y=negative_y_trace,name=model_2_trace_label,mode="lines",marker=dict(line=dict(color=model_2_line_color,width=2))))

    fig.update_layout(title=title_string,xaxis_title=x_axis_title_string,yaxis_title=y_axis_title_string,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return fig



def ranks_to_medal_string(value):
    """converts medal numeric encoding to string

    Args:
        value (int): value of medal indicator variable 

    Returns:
        string: returns correspoding medal type
    """
    if value == 1:
        output = 'Gold'
    elif value == 2:
        output = 'Silver'
    elif value == 3:
        output = 'Bronze'
    else:
        output = 'Non-Podium'
    return output


def Generate_Event_Violin_Plot(df,events,genders):
    """Generate Event level violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on

    Returns:
        Dash Figure: returns the event perspective dash figure given user selectoins
    """
    event_df = df.copy()
    event_df = Get_Specific_Group_Dataframe(df=event_df,events=events,genders=genders,min_rank=None,max_rank=None,class_name='Elite')
    event_df['Age Years'] = event_df.apply(lambda x: x['Athlete Age Days Derived']/365.25,axis=1)
    event_df['Type'] = event_df.apply(lambda x: ranks_to_medal_string(x['Adj_Rank']),axis=1)
    fig = go.Figure()
    if len(genders) > 1:
        grouped_labels_women = event_df['Event Name Short'][event_df['Event Gender'] == 'Women']
        grouped_labels_men = event_df['Event Name Short'][event_df['Event Gender'] == 'Men']
        fig.add_trace(go.Violin(x=[grouped_labels_women,event_df['Type'][event_df['Event Gender'] == 'Women']],
                                y=event_df['Age Years'][ event_df['Event Gender'] == 'Women' ],
                                legendgroup='Women', name='Women',scalemode='width',
                                line_color='rgba(245, 40, 145, 0.8)',side='positive',points='outliers',spanmode='hard'))
        fig.add_trace(go.Violin(x=[grouped_labels_men,event_df['Type'][event_df['Event Gender'] == 'Men']],
                                y=event_df['Age Years'][ event_df['Event Gender'] == 'Men' ],
                                legendgroup='Men', name='Men',scalemode='width',
                                line_color='rgba(39, 139, 245, 0.8)',side='negative',points='outliers',spanmode='hard'))
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    else:
        fig.add_trace(go.Violin(x=event_df['Event Name Short'][ event_df['Type'] == 'Non-Podium' ],
                        y=event_df['Age Years'][ event_df['Type'] == 'Non-Podium' ],
                        legendgroup='Non-Podium', name='Non-Podium',
                        line_color='rgba(61, 143, 255, 0.8)'))
        fig.add_trace(go.Violin(x=event_df['Event Name Short'][ event_df['Type'] == 'Gold' ],
                            y=event_df['Age Years'][ event_df['Type'] == 'Gold' ],
                            legendgroup='Gold',  name='Gold',
                            line_color='rgba(255, 184, 0, 0.8)'))
        fig.add_trace(go.Violin(x=event_df['Event Name Short'][ event_df['Type'] == 'Silver' ],
                            y=event_df['Age Years'][ event_df['Type'] == 'Silver' ],
                            legendgroup='Silver',  name='Silver',
                            line_color='rgba(163, 167, 167, 0.8)'))
        fig.add_trace(go.Violin(x=event_df['Event Name Short'][ event_df['Type'] == 'Bronze' ],
                            y=event_df['Age Years'][ event_df['Type'] == 'Bronze' ],
                            legendgroup='Bronze', name='Bronze',
                            line_color='rgba(128, 42, 0, 0.8)'))
    

        fig.update_traces(box_visible=True, meanline_visible=True)
        fig.update_layout(violinmode='group',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    return fig


def Generate_Indv_Rank_Violin_Plot(df,events,genders):
    """Generate indv rank level violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on

    Returns:
        Dash Figure: returns the indv rank perspective dash figure given user selectoins
    """
    event_df = df.copy()
    fig = go.Figure()
    if len(genders)>1:
        wr_scaled_grouped = Get_Rank_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Scaled_Rank')
        wr_adj_grouped = Get_Rank_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Adj_Rank')
        wr_concat = pd.concat([wr_adj_grouped, wr_scaled_grouped], ignore_index=True)
        grouped_labels_women = wr_concat['Event Name Short'][wr_concat['Event Gender'] == 'Women']
        grouped_labels_men = wr_concat['Event Name Short'][wr_concat['Event Gender'] == 'Men']
        fig.add_trace(go.Violin(x=[grouped_labels_women,wr_concat['Type'][wr_concat['Event Gender'] == 'Women']],
                                y=wr_concat['Age Years'][ wr_concat['Event Gender'] == 'Women' ],
                                legendgroup='Women', name='Women',
                                line_color='rgba(245, 40, 145, 0.8)',side='positive'))
        fig.add_trace(go.Violin(x=[grouped_labels_men,wr_concat['Type'][wr_concat['Event Gender'] == 'Men']],
                                y=wr_concat['Age Years'][ wr_concat['Event Gender'] == 'Men' ],
                                legendgroup='Men', name='Men',
                                line_color='rgba(39, 139, 245, 0.8)',side='negative'))
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    else:
        wr_scaled_grouped = Get_Rank_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Scaled_Rank')
        wr_adj_grouped = Get_Rank_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Adj_Rank')
        wr_concat = pd.concat([wr_adj_grouped, wr_scaled_grouped], ignore_index=True)
        fig.add_trace(go.Violin(x=wr_concat['Event Name Short'][ wr_concat['Type'] == 'Scaled_Rank' ],
                        y=wr_concat['Age Years'][ wr_concat['Type'] == 'Scaled_Rank' ],
                        legendgroup='Scaled Rank',  name='Scaled Rank',
                        line_color='#0b285f'))
        fig.add_trace(go.Violin(x=wr_concat['Event Name Short'][ wr_concat['Type'] == 'Adj_Rank' ],
                            y=wr_concat['Age Years'][ wr_concat['Type'] == 'Adj_Rank' ],
                            legendgroup='Rank',  name='Rank',
                            line_color='#c42032'))
        fig.update_traces(box_visible=True, meanline_visible=True)
        fig.update_layout(violinmode='group',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        
    return fig
    

def Generate_Indv_World_Rank_Violin_Plot(df,events,genders):
    """Generate indv world rank level violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on

    Returns:
        Dash Figure: returns the indv world rank perspective dash figure given user selections
    """
    event_df = df.copy()
    fig = go.Figure()
    if len(genders)>1:
        wr_scaled_grouped = Get_WR_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Scaled_Rank')
        wr_adj_grouped = Get_WR_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Adj_Rank')
        wr_concat = pd.concat([wr_adj_grouped, wr_scaled_grouped], ignore_index=True)
        grouped_labels_women = wr_concat['Event Name Short'][wr_concat['Event Gender'] == 'Women']
        grouped_labels_men = wr_concat['Event Name Short'][wr_concat['Event Gender'] == 'Men']
        fig.add_trace(go.Violin(x=[grouped_labels_women,wr_concat['Type'][wr_concat['Event Gender'] == 'Women']],
                                y=wr_concat['Age Years'][ wr_concat['Event Gender'] == 'Women' ],
                                legendgroup='Women', name='Women',
                                line_color='rgba(245, 40, 145, 0.8)',side='positive'))
        fig.add_trace(go.Violin(x=[grouped_labels_men,wr_concat['Type'][wr_concat['Event Gender'] == 'Men']],
                                y=wr_concat['Age Years'][ wr_concat['Event Gender'] == 'Men' ],
                                legendgroup='Men', name='Men',
                                line_color='rgba(39, 139, 245, 0.8)',side='negative'))
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    else:
        wr_scaled_grouped = Get_WR_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Scaled_Rank')
        wr_adj_grouped = Get_WR_Violin_Data(df=event_df,genders=genders,events=events,rank_type='Adj_Rank')
        wr_concat = pd.concat([wr_adj_grouped, wr_scaled_grouped], ignore_index=True)
        fig.add_trace(go.Violin(x=wr_concat['Event Name Short'][ wr_concat['Type'] == 'Scaled_Rank' ],
                        y=wr_concat['Age Years'][ wr_concat['Type'] == 'Scaled_Rank' ],
                        legendgroup='Scaled Rank', name='Scaled Rank',
                        line_color='#0b285f'))
        fig.add_trace(go.Violin(x=wr_concat['Event Name Short'][ wr_concat['Type'] == 'Adj_Rank' ],
                            y=wr_concat['Age Years'][ wr_concat['Type'] == 'Adj_Rank' ],
                            legendgroup='Rank', name='Rank',
                            line_color='#c42032'))
        fig.update_traces(box_visible=True, meanline_visible=True)
        fig.update_layout(violinmode='group',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        
    return fig

def Get_WR_Violin_Data(df,genders,events,rank_type):
    """Gets and returns formated alpine data for use in the indv world rank perspective violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type(str): whether to use scaled or adjusted rank

    Returns:
       dataframe: formatted data for input into violin indv world rank plot function
    """
    df_copy = df.copy()
    wr_df = Get_WR_DF(df=df_copy,genders=genders,events=events,class_name='Elite')
    wr_grouped = wr_df.groupby(by=['Person ID','Event Name Short','Event Gender'])[rank_type].min().reset_index()
    person_id_array = np.array(wr_grouped['Person ID'])
    event_array = np.array(wr_grouped['Event Name Short'])
    gender_array = np.array(wr_grouped['Event Gender'])
    best_rank_achieved = np.array(wr_grouped[rank_type])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_i = person_id_array[i]
        event_i = event_array[i]
        gender_i = gender_array[i]
        best_rank_i = best_rank_achieved[i]
        wr_df_copy = wr_df.copy()
        wr_df_copy = wr_df_copy[(wr_df_copy['Person ID'] == person_i)&(wr_df_copy[rank_type] == best_rank_i)&(wr_df_copy['Event Name Short'] == event_i)&(wr_df_copy['Event Gender'] == gender_i)]
        age_list.append(wr_df_copy['Athlete Age Days Derived'].mean())
    age_array = np.array(age_list)/365.25
    type_list = [rank_type]*age_array.shape[0]
    type_array = np.array(type_list)
    wr_array_joined = np.concatenate((person_id_array[:,None],event_array[:,None],gender_array[:,None],age_array[:,None],type_array[:,None]),axis=1)
    wr_df_out = pd.DataFrame(wr_array_joined, columns = ['Person ID','Event Name Short','Event Gender','Age Years','Type'])
    return wr_df_out


def Get_Rank_Violin_Data(df,genders,events,rank_type):
    """Gets and returns formated alpine data for use in the indv rank perspective violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on
         rank_type(str): whether to use scaled or adjusted rank

    Returns:
        dataframe: formatted data for input into violin indv rank plot function
    """
    df_copy = df.copy()
    filtered_timed_events = Get_Timed_Events(df=df_copy,genders=genders,events=events,class_name='Elite',include_rankings=False)
    filtered_timed_events = Filter_Timed_Events(filtered_timed_events,min_entries=6)
    filtered_timed_events = filtered_timed_events.dropna()
    filtered_timed_events = filtered_timed_events[['Person ID','Scaled_Rank','Athlete Age Days Derived','Adj_Rank','Event Name Short','Event Gender']]
    grouped_rank = filtered_timed_events.groupby(by=['Person ID','Event Name Short','Event Gender'])[rank_type].min().reset_index()
    rank_array = np.array(grouped_rank[rank_type])
    person_id_array = np.array(grouped_rank['Person ID'])
    event_array = np.array(grouped_rank['Event Name Short'])
    gender_array = np.array(grouped_rank['Event Gender'])
    age_list = []
    for i in range(0,person_id_array.shape[0]):
        person_id_i = person_id_array[i]
        event_i = event_array[i]
        gender_i = gender_array[i]
        rank_i = rank_array[i]
        athlete_i_df = filtered_timed_events[(filtered_timed_events['Person ID']==person_id_i)&(filtered_timed_events[rank_type]==rank_i)&(filtered_timed_events['Event Name Short']==event_i)&(filtered_timed_events['Event Gender']==gender_i)]
        age_list.append(athlete_i_df['Athlete Age Days Derived'].min())
    age_array = np.array(age_list)/365.25
    type_list = [rank_type]*age_array.shape[0]
    type_array = np.array(type_list)
    rank_array_joined = np.concatenate((person_id_array[:,None],event_array[:,None],gender_array[:,None],age_array[:,None],type_array[:,None]),axis=1)
    rank_df_out = pd.DataFrame(rank_array_joined, columns = ['Person ID','Event Name Short','Event Gender','Age Years','Type'])
    return rank_df_out

def Generate_Reg_Violin_Plot(df,events,genders):
    """Generate regulation level violin plot

    Args:
        df(dataframe): alpine events dataframe
         genders(list or string): genders to filter on
         events(list or string): list of event name short strings to filter on

    Returns:
        Dash Figure: returns the regulation perspective dash figure given user selectoins
    """
    event_df_pre = Get_Specific_Group_Dataframe(df=df,events=events,genders=genders,min_rank=None,max_rank=None,class_name='Elite',filter_reg=True,reg_type='pre')
    event_df_pre['Type'] = event_df_pre.apply(lambda x: 'Pre Regulation',axis=1) 
    event_df_post = Get_Specific_Group_Dataframe(df=df,events=events,genders=genders,min_rank=None,max_rank=None,class_name='Elite',filter_reg=True,reg_type='post')
    event_df_post['Type'] = event_df_post.apply(lambda x: 'Post Regulation',axis=1)
    event_df = pd.concat([event_df_pre, event_df_post], ignore_index=True)
    event_df['Age Years'] = event_df.apply(lambda x: x['Athlete Age Days Derived']/365.25,axis=1)
    fig = go.Figure()
    if len(genders) > 1:
        grouped_labels_women = event_df['Event Name Short'][event_df['Event Gender'] == 'Women']
        grouped_labels_men = event_df['Event Name Short'][event_df['Event Gender'] == 'Men']
        fig.add_trace(go.Violin(x=[grouped_labels_women,event_df['Type'][event_df['Event Gender'] == 'Women']],
                                y=event_df['Age Years'][ event_df['Event Gender'] == 'Women' ],
                                legendgroup='Women', name='Women',
                                line_color='rgba(245, 40, 145, 0.8)',side='positive'))
        fig.add_trace(go.Violin(x=[grouped_labels_men,event_df['Type'][event_df['Event Gender'] == 'Men']],
                                y=event_df['Age Years'][ event_df['Event Gender'] == 'Men' ],
                                legendgroup='Men', name='Men',
                                line_color='rgba(39, 139, 245, 0.8)',side='negative'))
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    else:
        event_df['Medal Type'] = event_df.apply(lambda x: ranks_to_medal_string(x['Adj_Rank']),axis=1)
        grouped_labels_pre = event_df['Event Name Short'][event_df['Type'] == 'Pre Regulation']
        grouped_labels_post = event_df['Event Name Short'][event_df['Type'] == 'Post Regulation']
        fig.add_trace(go.Violin(x=[grouped_labels_post,event_df['Type'][ (event_df['Medal Type'] == 'Non-Podium')&(event_df['Type'] == 'Post Regulation') ]],
                        y=event_df['Age Years'][ (event_df['Medal Type'] == 'Non-Podium')&(event_df['Type'] == 'Post Regulation') ],
                        legendgroup='Post Regulation', name = 'Non-Podium', legendgrouptitle_text='Post Regulation',
                        line_color='rgba(61, 143, 255, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_post,event_df['Type'][ (event_df['Medal Type'] == 'Gold')&(event_df['Type'] == 'Post Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Gold')&(event_df['Type'] == 'Post Regulation') ],
                            legendgroup='Post Regulation',  name='Post Regulation: Gold',
                            line_color='rgba(255, 184, 0, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_post,event_df['Type'][ (event_df['Medal Type'] == 'Silver')&(event_df['Type'] == 'Post Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Silver')&(event_df['Type'] == 'Post Regulation') ],
                            legendgroup='Post Regulation', name='Post Regulation: Silver',
                            line_color='rgba(163, 167, 167, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_post,event_df['Type'][ (event_df['Medal Type'] == 'Bronze')&(event_df['Type'] == 'Post Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Bronze')&(event_df['Type'] == 'Post Regulation') ],
                            legendgroup='Post Regulation',  name='Post Regulation: Bronze',
                            line_color='rgba(128, 42, 0, 0.8)'))

        fig.add_trace(go.Violin(x=[grouped_labels_pre,event_df['Type'][ (event_df['Medal Type'] == 'Non-Podium')&(event_df['Type'] == 'Pre Regulation') ]],
                        y=event_df['Age Years'][ (event_df['Medal Type'] == 'Non-Podium')&(event_df['Type'] == 'Pre Regulation') ],
                        legendgroup='Pre Regulation',  name='Pre Regulation: Non-Podium',legendgrouptitle_text='Pre Regulation',
                        line_color='rgba(61, 143, 255, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_pre,event_df['Type'][ (event_df['Medal Type'] == 'Gold')&(event_df['Type'] == 'Pre Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Gold')&(event_df['Type'] == 'Pre Regulation') ],
                            legendgroup='Pre Regulation', name='Pre Regulation: Gold',
                            line_color='rgba(255, 184, 0, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_pre,event_df['Type'][ (event_df['Medal Type'] == 'Silver')&(event_df['Type'] == 'Pre Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Silver') &(event_df['Type'] == 'Pre Regulation')],
                            legendgroup='Pre Regulation', name='Pre Regulation: Silver',
                            line_color='rgba(163, 167, 167, 0.8)'))
        fig.add_trace(go.Violin(x=[grouped_labels_pre,event_df['Type'][ (event_df['Medal Type'] == 'Bronze')&(event_df['Type'] == 'Pre Regulation') ]],
                            y=event_df['Age Years'][ (event_df['Medal Type'] == 'Bronze')&(event_df['Type'] == 'Pre Regulation') ],
                            legendgroup='Pre Regulation', name='Pre Regulation: Bronze',
                            line_color='rgba(128, 42, 0, 0.8)'))
    

        fig.update_traces(box_visible=True, meanline_visible=True)
        fig.update_layout(violinmode='group',title='Violin Plot of Event & Gender Performance',yaxis_title='Age in Years at Achievement',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    return fig

def Percentile_Peak_Age_Range_Bounds(percentile_range,samples,scores):
    """Generates the subset x and y observations from a trace to integrate the percentile range over

    Args:
        prcentile_range: range of decimals to integrate from distribtion mode
        samples: kde sampeles to operate over
        scores: kde scores to operate over

    Returns:
        np.array: start point of percentile integration
        np.array: end point of percentile integration
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
        
    return x_trace[0],x_trace[-1]


def Get_Event_Gender_Peak_Age_Bounds():
    """Generates age dictionary encoding for the optimal event kde age ranges

    Args:
        function takes no arguments

    Returns:
        dict: event perspective optimal age bounds
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

def Get_Indv_Rank_Gender_Peak_Age_Bounds():
    """Generates age dictionary encoding for the optimal indv rank kde age ranges

    Args:
        function takes no arguments

    Returns:
        dict: indv rank perspective optimal age bounds
    """
    df = pd.read_csv(DATA_PATH.joinpath('Modeling Results/INDV_KDE_RESULT_DF.csv'))
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
            events_j_dict['Age Lower'] = filterd_df_ij['Unscaled Age Lower'].iloc()[0]
            events_j_dict['Age Upper'] = filterd_df_ij['Unscaled Age Upper'].iloc()[0]
            gender_dict_i[event_j] = events_j_dict
        age_dict[gender_i] = gender_dict_i
    return age_dict

def Get_Indv_WR_Gender_Peak_Age_Bounds():
    """Generates age dictionary encoding for the optimal indv wr kde age ranges

    Args:
        function takes no arguments

    Returns:
        dict: indv wr rank perspective optimal age bounds
    """
    df = pd.read_csv(DATA_PATH.joinpath('Modeling Results/INDV_WR_KDE_RESULT_DF.csv'))
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
            events_j_dict['Age Lower'] = filterd_df_ij['Unscaled Age Lower'].iloc()[0]
            events_j_dict['Age Upper'] = filterd_df_ij['Unscaled Age Upper'].iloc()[0]
            gender_dict_i[event_j] = events_j_dict
        age_dict[gender_i] = gender_dict_i
    return age_dict


def Generate_KDE_Age_Range_Results_Table(mode=1):
    """Gets and converts the identified kde methods corresponding optimal age ranges into an html table

    Args:
        mode (int): kde method from which to pull the age ranges from

    Returns:
        dash div component: dash div containing the age range tables
    """
    if mode == 1:
        peak_age_bounds_dict = Get_Event_Gender_Peak_Age_Bounds()
    elif mode == 2:
        peak_age_bounds_dict = Get_Indv_Rank_Gender_Peak_Age_Bounds()
    else:
        peak_age_bounds_dict = Get_Indv_WR_Gender_Peak_Age_Bounds()

    unique_events = ['All','Combination','Downhill','Giant Slalom','Slalom','Super G']
    unique_genders = ['All','Men','Women']
    gender_col = []
    event_col = []
    age_lower_col = []
    age_upper_col = []
    for i in range(0,len(unique_genders)):
        gender_i = unique_genders[i]
        for j in range(0,len(unique_events)):
            event_j = unique_events[j]
            gender_col.append(gender_i)
            event_col.append(event_j)
            age_lower_col.append(peak_age_bounds_dict[gender_i][event_j]['Age Lower'])
            age_upper_col.append(peak_age_bounds_dict[gender_i][event_j]['Age Upper'])

    data_list = np.array([gender_col,event_col,age_lower_col,age_upper_col]).T
    data_columns = ['Gender','Event','Peak Lower','Peak Upper']
    age_df = pd.DataFrame(data_list)
    age_df.columns = data_columns
    age_df['KDE Identified Peak Age Range'] = age_df.apply(lambda x: html.Div([dcc.RangeSlider(20,30,0.01,value=[float(x['Peak Lower']),float(x['Peak Upper'])],marks=None,disabled=True,tooltip={"placement": "bottom","always_visible":True},className='age-range-slider')],className='dash-container mwpx300 shelf'),axis=1)
    age_df_index_cols = ['Gender','Event']
    age_df = age_df[['Gender','Event','KDE Identified Peak Age Range']]
    all_table =  Multi_Index_DataFrame_To_HTML_Table(age_df[age_df['Gender']=='All'],index_cols=age_df_index_cols,classNameRoot='dash-df')
    men_table = Multi_Index_DataFrame_To_HTML_Table(age_df[age_df['Gender']=='Men'],index_cols=age_df_index_cols,classNameRoot='dash-df')
    women_table = Multi_Index_DataFrame_To_HTML_Table(age_df[age_df['Gender']=='Women'],index_cols=age_df_index_cols,classNameRoot='dash-df')
    
    return [all_table,men_table,women_table]
