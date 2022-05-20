import numpy as np
from scipy.interpolate import BSpline
from scipy.interpolate import splrep
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


#mean squared error based on an obervation array and true label array
def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def cv_split_sequential_data(X,Y,n_folds=5,seed=None):
    
    '''Takes as input paired sequential data of dim (n_observations,)
    Returns a shuffled array that preserves original x and y pairing
    of shape (number of folds, number of observations // by number of folds)'''
    
    if seed is not None:
        np.random.seed(seed)
    
    data_array = np.vstack((Y,X)).T
    np.random.shuffle(data_array)
    data_array = data_array.T
    n_obs = data_array.shape[1]
    samples_per_fold = n_obs//n_folds
    CV_X_Array = np.zeros((n_folds,samples_per_fold))
    CV_Y_Array = np.zeros((n_folds,samples_per_fold))
    for i in range(0,n_folds):
        start_index = i*samples_per_fold
        end_index = (i+1)*samples_per_fold
        CV_X_Array[i,:] = data_array[1,start_index:end_index]
        CV_Y_Array[i,:] = data_array[0,start_index:end_index]
    return CV_Y_Array,CV_X_Array




class Spline():
    
    def __init__(self,X,Y,knots,order=4):
        
        self.order = order
        self.initial_data = X
        self.initial_labels = Y
        
        self.data = None
        self.labels = None
        
        self.knot_start = None
        self.knot_end = None
        
        self.user_knots = knots
        self.knots = None
        self.num_knots = None
        
        self.H_train = None
        self.B = None

        
    def sort_xy(self):
        sort_idxs = np.argsort(self.initial_data,axis=0)
        self.data = np.take_along_axis(self.initial_data,sort_idxs,axis=0)
        self.labels = np.take_along_axis(self.initial_labels,sort_idxs,axis=0)
        self.knot_start = self.data[0]
        self.knot_end = self.data[-1]

    def initialize_knots(self):
        if type(self.user_knots) == int:
            self.num_knots = self.user_knots
            self.knots = np.linspace(self.knot_start,self.knot_end,self.user_knots)
        else:
            self.knots = self.user_knots
            self.num_knots = len(self.user_knots)

    def form_H(self,X):

        H = np.ones((X.shape[0],self.order+self.num_knots))

        for i in range(0,self.order):
            H[:,i] = X**i

        for j in range(0,self.num_knots):
            H[:,j+self.order] = np.maximum((X - self.knots[j])**(self.order-1),0)
        return H

    def solve_B(self):
        self.B = np.linalg.lstsq(self.H_train, self.labels,rcond=-1)[0]

    def fit(self):
        self.sort_xy()
        self.initialize_knots()
        self.H_train = self.form_H(self.data)
        self.solve_B()
        self.trained_model = self.H_train@self.B

    def predict(self,X_test):
        X_test = np.sort(X_test)
        test_H = self.form_H(X_test)
        return test_H@self.B    
    


class B_Spline():
    
    def __init__(self,X,Y,knots,order=4):
        self.order = order
        self.degree = self.order-1
        
        self.initial_data = X
        self.initial_labels = Y
        
        self.data = None
        self.labels = None
        
        self.knot_start = None
        self.knot_end = None
        
        self.user_knots = knots
        self.knots = None
        self.num_knots = None
        
        self.fit_spline = None


        
    def sort_xy(self):
        sort_idxs = np.argsort(self.initial_data,axis=0)
        self.data = np.take_along_axis(self.initial_data,sort_idxs,axis=0)
        self.labels = np.take_along_axis(self.initial_labels,sort_idxs,axis=0)
        self.knot_start = self.data[1]
        self.knot_end = self.data[-2]

    def initialize_knots(self):
        if type(self.user_knots) == int:
            self.num_knots = self.user_knots
            knots = np.linspace(self.knot_start,self.knot_end,self.user_knots+2)
            knots = knots[1:]
            self.knots = knots[:-1]
        else:
            self.knots = self.user_knots
            self.num_knots = len(self.user_knots)
            

    def form_spline_basis(self):
        t, c, k = interpolate.splrep(x=self.data,y=self.labels,k=self.order-1,t=self.knots)
        self.fit_spline = interpolate.BSpline(t, c, k, extrapolate=True) 
        
    def fit(self):
        self.sort_xy()
        self.initialize_knots()
        self.form_spline_basis()


    def predict(self,X_test):
        X_test = np.sort(X_test)
        
        return self.fit_spline(X_test)

class Smoothing_Spline():
    
    def __init__(self,X,Y,knots,smoothing=1e-3,order=4):
        self.order = order
        self.smooth = smoothing
        self.degree = self.order-1
        
        self.initial_data = X
        self.initial_labels = Y
        
        self.data = None
        self.labels = None
        
        self.knot_start = None
        self.knot_end = None
        
        self.user_knots = knots
        self.knots = None
        self.num_knots = None
        self.num_unique_knots = None
        self.num_interior_knots = None
        
        self.fit_spline_basis = None
        self.B = None
        self.deg_of_freedom = None
        self.S_matrix = None
        

    def sort_xy(self):
        sort_idxs = np.argsort(self.initial_data,axis=0)
        self.data = np.take_along_axis(self.initial_data,sort_idxs,axis=0)
        self.labels = np.take_along_axis(self.initial_labels,sort_idxs,axis=0)
        self.knot_start = self.data[0]
        self.knot_end = self.data[-1]

    def initialize_knots(self):
        if type(self.user_knots) == int:
            self.num_knots = self.user_knots
            knot_vector = np.linspace(self.knot_start,self.knot_end,self.num_knots)
            num_duplicate_knots = self.order-1
            knot_vector = np.insert(knot_vector,0,[self.knot_start]*num_duplicate_knots)
            knot_vector = np.append(knot_vector,[self.knot_end]*num_duplicate_knots)
            self.knots = knot_vector
            self.num_unique_knots = self.num_knots
            self.num_interior_knots = self.num_knots-2
            self.deg_of_freedom = self.num_unique_knots + self.order

            

    def form_spline_basis(self):
        self.fit_spline_basis = BSpline(self.knots, np.eye(self.deg_of_freedom),self.order-1,extrapolate=True)
        

    def solve_B(self,X):
        B = self.fit_spline_basis(X)
        return B[:,:-2]
     
    def solve_B2(self,B):
        return np.diff(B,axis=0,n=2)*(self.data.shape[0]-1)**2 
     
    def solve_omega(self,B2):
        return B2.T.dot(B2)/(self.data.shape[0]-2)
        
    def fit(self):
        self.sort_xy()
        self.initialize_knots()
        self.form_spline_basis()
        self.B = self.solve_B(self.data)
        self.B2 = self.solve_B2(self.B)
        self.omega = self.solve_omega(self.B2)
        self.S_matrix = self.B@np.linalg.pinv(self.B.T@self.B+self.smooth*self.omega)@self.B.T 
        self.trained_model = self.S_matrix.dot(self.labels)
        

    def predict(self,X_test,smooth,return_trace=False):
        X_test = np.sort(X_test)
        pred_B = self.solve_B(X_test)
        pred_B2 = self.solve_B2(pred_B)
        pred_omega = self.solve_omega(pred_B2)
        S_pred = pred_B@np.linalg.pinv(pred_B.T@pred_B+smooth*pred_omega)@pred_B.T 
        if return_trace:
            return S_pred.dot(self.labels), np.trace(S_pred)
        else:
            return S_pred.dot(self.labels)
        
        
class Kernel_Smoother():
    
    def __init__(self,X,Y,bandwidth=1):
        self.data = X
        self.labels = Y
        self.bandwidth = bandwidth
        
    def Gaussian_Kernel(self,u):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-u ** 2)

    def predict_point(self,x):
        z = self.Gaussian_Kernel((x - self.data)/self.bandwidth)
        y_hat = np.average(self.labels,weights=z)
        return y_hat
    
    def predict(self,x_test):
        preds = np.ones((x_test.shape[0],))
        for i in range(0,preds.shape[0]):
            preds[i] = self.predict_point(x_test[i])
        return preds
    
    
def run_cubic_spline_cv(X,Y,n_folds=5,min_knots=5,max_knots=50,seed=None,order=3):

    CV_Y,CV_X = cv_split_sequential_data(X,Y,n_folds=n_folds,seed=seed)
    
    knot_mse = []
    knots_range = []
    
    
    for k in range(min_knots,max_knots+1):
        
        knot_mse_list = []
        knots_range.append(k)
        
        for n in range(0,n_folds):
            mask = [True] * n_folds
            mask[n] = False
            Y_train = CV_Y[mask,:].reshape((CV_Y.shape[1]*(n_folds-1),))
            X_train = CV_X[mask,:].reshape((CV_X.shape[1]*(n_folds-1),))
            
            Y_test = CV_Y[n,:]
            X_test = CV_X[n,:]
            
            test_sort_idxs = np.argsort(X_test,axis=0)
            X_test = np.take_along_axis(X_test,test_sort_idxs,axis=0)
            Y_test = np.take_along_axis(Y_test,test_sort_idxs,axis=0)
            
            kn_spline = Spline(X=X_train,Y=Y_train,knots=k,order=order)
            kn_spline.fit()
            
            Y_hat = kn_spline.predict(X_test)
            knot_mse_list.append(MSE(Y_test,Y_hat))
        
        knot_mse.append(np.mean(knot_mse_list))
        
    output_dict = dict()
    
    output_dict['best_hyperparameter'] = knots_range[np.argmin(knot_mse)]
    output_dict['best_model_performance'] = np.min(knot_mse)
    output_dict['performance_by_hyperparameter'] = knot_mse
    output_dict['hyperparameter_range'] = knots_range
    return output_dict
    
    
    
def run_b_spline_cv(X,Y,n_folds=5,min_knots=5,max_knots=50,seed=None,order=3):

    CV_Y,CV_X = cv_split_sequential_data(X,Y,n_folds=n_folds,seed=seed)
    knot_mse = []
    knots_range = []
    
    
    for k in range(min_knots,max_knots+1):
        
        knot_mse_list = []
        knots_range.append(k)
        
        for n in range(0,n_folds):
            mask = [True] * n_folds
            mask[n] = False
            Y_train = CV_Y[mask,:].reshape((CV_Y.shape[1]*(n_folds-1),))
            X_train = CV_X[mask,:].reshape((CV_X.shape[1]*(n_folds-1),))
            
            Y_test = CV_Y[n,:]
            X_test = CV_X[n,:]
            
            test_sort_idxs = np.argsort(X_test,axis=0)
            X_test = np.take_along_axis(X_test,test_sort_idxs,axis=0)
            Y_test = np.take_along_axis(Y_test,test_sort_idxs,axis=0)
            kn_spline = B_Spline(X=X_train,Y=Y_train,order=order,knots=k)
            kn_spline.fit()
            Y_hat = kn_spline.predict(X_test)
            knot_mse_list.append(MSE(Y_test,Y_hat))
        knot_mse.append(np.mean(knot_mse_list))

    output_dict = dict()
    output_dict['best_hyperparameter'] = knots_range[np.argmin(knot_mse)]
    output_dict['best_model_performance'] = np.min(knot_mse)
    output_dict['performance_by_hyperparameter'] = knot_mse
    output_dict['hyperparameter_range'] = knots_range
    
    
    return output_dict
    
    
def run_gcv_smoothing_spline(X,Y,order=3,smoothing_vals = np.arange(0.0000001, 0.000001,0.00000001)):
    k=X.shape[0]
    smoothing_spline = Smoothing_Spline(X,Y,knots=k,smoothing=1e-3,order=order)
    smoothing_spline.fit()
    df_array = np.ones((smoothing_vals.shape[0],))
    RSS_array = np.ones((smoothing_vals.shape[0],))
    for i in range(0,smoothing_vals.shape[0]):
        preds,df = smoothing_spline.predict(X,smoothing_vals[i],return_trace=True)
        RSS_array[i] = ((Y-preds)**2).sum()
        df_array[i] = df
    
    GCV = (RSS_array/k)/(1-df_array/k)**2
    optimal_index = np.argmin(GCV)
    optimal_lambda = smoothing_vals[optimal_index]
    optimal_loss = GCV[optimal_index]
    
    smoothing_spline = Smoothing_Spline(X,Y,knots=k,smoothing=optimal_lambda,order=order)
    smoothing_spline.fit()
    preds = smoothing_spline.predict(X,optimal_lambda)
    optimal_model_mse = MSE(preds,Y)
    output_dict = dict()
    
    output_dict['best_model_lambda'] = optimal_lambda
    output_dict['best_GCV'] = optimal_loss
    output_dict['performance_by_hyperparameter'] = GCV
    output_dict['hyperparameter_values'] = smoothing_vals
    output_dict['optimal_model_mse'] = optimal_model_mse
    output_dict['optimal_model_preds'] = preds
    
    return output_dict
    
def run_kernel_cv(X,Y,n_folds=5,seed=None,bandwidths=np.arange(0.00009,0.01,0.0001)):

    CV_Y,CV_X = cv_split_sequential_data(X,Y,n_folds=n_folds,seed=seed)
    
    global_bandwidth_mse_list = []
    
    for k in range(0,len(bandwidths)):
        
        bandwidth_mse_list = []

        for n in range(0,n_folds):
            mask = [True] * n_folds
            mask[n] = False
            Y_train = CV_Y[mask,:].reshape((CV_Y.shape[1]*(n_folds-1),))
            X_train = CV_X[mask,:].reshape((CV_X.shape[1]*(n_folds-1),))
            
            Y_test = CV_Y[n,:]
            X_test = CV_X[n,:]
            
            test_sort_idxs = np.argsort(X_test,axis=0)
            X_test = np.take_along_axis(X_test,test_sort_idxs,axis=0)
            Y_test = np.take_along_axis(Y_test,test_sort_idxs,axis=0)
            kern = Kernel_Smoother(X=X_train,Y=Y_train,bandwidth=bandwidths[k])
            Y_hat = kern.predict(X_test)
            bandwidth_mse_list.append(MSE(Y_test,Y_hat))
        global_bandwidth_mse_list.append(np.mean(bandwidth_mse_list))

    output_dict = dict()
    output_dict['best_hyperparameter'] = bandwidths[np.argmin(global_bandwidth_mse_list)]
    output_dict['best_model_performance'] = np.min(global_bandwidth_mse_list) 
    output_dict['performance_by_hyperparameter'] = global_bandwidth_mse_list
    output_dict['hyperparameter_range'] = bandwidths
    
    
    return output_dict

    