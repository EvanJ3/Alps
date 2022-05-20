import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
pd.options.mode.chained_assignment = None
from sklearn.model_selection import GridSearchCV
import pathlib
from joblib import dump
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()
MODEL_PATH = PATH.joinpath("../../Models").resolve()


##############################################################
# Calculates mean squared error given a prediction and label vectors
# ############################################################
def MSE(preds,labels):
    return np.mean((preds-labels)**2.0)


##############################################################
# generate and format a dataframe into the representtion 
# needed to train our machine leanring models
# ############################################################
def Generate_ML_Dataframe():
    olympic_results_df = pd.read_csv(DATA_PATH.joinpath('Olympic_Post_Hoc_Results.csv'))
    df_olympics = pd.read_csv(DATA_PATH.joinpath('Olympic_Ranking_Data.csv'))
    df_olympics_agg = df_olympics.groupby(by=['Olympics Name','NOC']).sum()[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']].reset_index()
    olympic_ml_df = df_olympics_agg.merge(olympic_results_df,how='left',left_on=['Olympics Name','NOC'],right_on=['Olympics Name','NOC'],validate='one_to_one')
    olympic_ml_df = olympic_ml_df.drop(columns=['Bronze','Silver','Gold','Olympic Rank'])
    olympic_ml_df = olympic_ml_df.rename(columns={'Rank Score':'Medal Score'})
    olympic_ml_df = olympic_ml_df.fillna(0.0)
    olympic_ml_df['Sum of Olympic Medal Score'] = olympic_ml_df.groupby(by=['Olympics Name'])['Medal Score'].transform('sum')
    olympic_ml_df.to_csv(MODEL_PATH.joinpath('ML_Input_Data.csv'),index=False)
    olympic_cv_df = olympic_ml_df[~olympic_ml_df['Olympics Name'].isin(['Milan Cortina 2026'])]
    X = np.array(olympic_cv_df[['Pre-Peak Age Athletes','Peak Age Athletes','Post-Peak Age Athletes']])
    y = np.array(olympic_cv_df['Medal Score'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    
    lin_reg_param_combinations = {'positive':[True,False],'normalize':[True,False],'fit_intercept':[True,False]}
    rt_param_combinations = {'max_depth':[3,5,10,15,20,25],'max_leaf_nodes':[3,5,10,15,20,25]}
    ridge_param_combinations = {'alpha':np.logspace(-1, .5, 25),'normalize':[True,False],'fit_intercept':[True,False]}
    lasso_param_combinations = {'alpha':np.logspace(-1, .5, 25),'normalize':[True,False],'fit_intercept':[True,False]}
    linear_reg_clf = LinearRegression()
    rt_clf = DecisionTreeRegressor()
    ridge_clf = Ridge()
    lasso_clf = Lasso()
    clf_dict = {
        "LINREG": {"model":linear_reg_clf,"cv_params":lin_reg_param_combinations},
        "RT": {"model":rt_clf,"cv_params":rt_param_combinations},
        "RIDGE": {"model":ridge_clf,"cv_params":ridge_param_combinations},
        "LASSO": {"model":lasso_clf,"cv_params":lasso_param_combinations}
    }
    results_list = []
    model_keys = list(clf_dict.keys())
    for i in range(0,len(model_keys)):
        key_i = model_keys[i]
        base_clf_i = clf_dict[key_i]["model"]
        clf_params_i = clf_dict[key_i]["cv_params"]
        grid_i = GridSearchCV(base_clf_i, clf_params_i,n_jobs=12,return_train_score=True,scoring='neg_mean_squared_error')
        grid_i.fit(X_train, y_train)
        results_i = pd.DataFrame(grid_i.cv_results_)
        results_i['Model Class'] = results_i.apply(lambda x:key_i,axis=1)
        estimator_name = key_i+'_optimal_medal_prediction_model.joblib'
        dump(grid_i.best_estimator_, MODEL_PATH.joinpath(estimator_name))
        results_list.append(results_i)
        
    cv_output_results_frame = pd.concat(results_list)
    cv_output_results_frame =cv_output_results_frame[['Model Class','param_fit_intercept','param_normalize','param_positive','mean_test_score','mean_train_score','param_max_depth','param_max_leaf_nodes','param_alpha']]
    cv_output_results_frame.to_csv(MODEL_PATH.joinpath('Medal_Prediction_CV_Results.csv'))
    
if __name__ == "__main__":
    Generate_ML_Dataframe()


