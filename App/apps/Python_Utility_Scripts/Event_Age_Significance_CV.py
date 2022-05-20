import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
import warnings
from scipy.stats import ks_2samp
import pathlib
import os

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))

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


def Percentile_Peak_Age_Range_Bounds(percentile_range,samples,scores):
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
        
    return x_trace[0],x_trace[-1]

def run():
    gender_options = ['Men','Women',['Men','Women']]
    event_options = ['Slalom','Giant Slalom','Super G','Downhill','Combination',['Slalom','Giant Slalom','Super G','Downhill','Combination']]
    result_df_column_names = ['KDE Method','Genders','Events','Full Optimal Kernel','Full Optimal BW','Top Optimal Kernel','Top Optimal BW','Full Age Lower','Full Age Upper','Top Age Lower','Top Age Upper','KS Test Statistic','KS P-Value']

    method_list = []
    gender_list = []
    event_list = []

    full_optimal_kernel = []
    full_optimal_bw = []

    top_optimal_kernel = []
    top_optimal_bw = []

    full_lower_age = []
    full_upper_age = []

    top_lower_age = []
    top_upper_age = []

    ks_stat_list = []
    ks_pvalue_list = []

    df_copy = df.copy()
    seed = 10
    for i in range(0,len(gender_options)):
        genders_i = gender_options[i]

        if type(genders_i) != list:
            genders_i = [genders_i]

        if genders_i == ['Men','Women']:
            gender_string_i = 'All'
        else:
            gender_string_i = ' '.join(genders_i)


        for j in range(0,len(event_options)):
            events_j = event_options[j]

            if type(events_j) != list:
                events_j = [events_j]

            if events_j == ['Slalom','Giant Slalom','Super G','Downhill','Combination']:
                events_j_string = 'All'
            else:
                events_j_string = ' '.join(events_j)
            print(f'Gender = {gender_string_i}')
            print(f'Event = {events_j_string}')
            Full_cv_results = Event_KDE_CV(df_copy,genders=genders_i,events=events_j,top_n=None)
            Full_optimal_kernel,Full_optimal_bw = Parse_Optimal_Parameters_From_CV_Results(Full_cv_results)

            Top_cv_results = Event_KDE_CV(df_copy,genders=genders_i,events=events_j,top_n=3)
            Top_optimal_kernel,Top_optimal_bw = Parse_Optimal_Parameters_From_CV_Results(Top_cv_results)

            Full_optimal_kde = Generate_Event_KDE_Fit(df=df_copy,genders=genders_i,events=events_j,mode=2,kernel=Full_optimal_kernel,bandwidth=Full_optimal_bw)
            Top_optimal_kde = Generate_Event_KDE_Fit(df=df_copy,genders=genders_i,events=events_j,mode=1,kernel=Top_optimal_kernel,bandwidth=Top_optimal_bw)

            Full_samples = Custom_Sample_KDE(Full_optimal_kde,n_samples=10000,random_state=seed)
            Full_scores = np.exp(Full_optimal_kde.score_samples(Full_samples))
            Full_lower_age,Full_upper_age = Percentile_Peak_Age_Range_Bounds([.5-(1/6),.5+(1/6)],samples=Full_samples,scores=Full_scores)


            Top_samples = Custom_Sample_KDE(Top_optimal_kde,n_samples=10000,random_state=seed)
            Top_scores = np.exp(Top_optimal_kde.score_samples(Top_samples))
            Top_lower_age,Top_upper_age = Percentile_Peak_Age_Range_Bounds([.5-(1/6),.5+(1/6)],samples=Top_samples,scores=Top_scores)
            test_stat_ij, pvalue_ij = ks_2samp(Full_samples.squeeze(), Top_samples.squeeze())
            method_list.append('Event KDE')
            gender_list.append(gender_string_i)
            event_list.append(events_j_string)

            full_optimal_kernel.append(Top_optimal_kernel)
            full_optimal_bw.append(Full_optimal_bw)

            top_optimal_kernel.append(Top_optimal_kernel)
            top_optimal_bw.append(Top_optimal_bw)

            full_lower_age.append(Full_lower_age)
            full_upper_age.append(Full_upper_age)

            top_lower_age.append(Top_lower_age)
            top_upper_age.append(Top_upper_age)

            ks_stat_list.append(test_stat_ij)
            ks_pvalue_list.append(pvalue_ij)
            print('--------------------------------')

    result_arrays = np.array([method_list,gender_list,event_list,full_optimal_kernel,full_optimal_bw,top_optimal_kernel,top_optimal_bw,full_lower_age,full_upper_age,top_lower_age,top_upper_age,ks_stat_list,ks_pvalue_list]).T
    result_df = pd.DataFrame(result_arrays,columns=result_df_column_names)

    result_df.to_csv(DATA_PATH.joinpath('KDE_RESULT_DF.csv'),index=False)
    
if __name__ == '__main__':
    run()