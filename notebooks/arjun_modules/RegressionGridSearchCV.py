import pandas as pd
from sklearn.metrics import mean_poisson_deviance,mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.model_selection import KFold
from datetime import datetime
import itertools
import warnings
import os.path

warnings.filterwarnings('ignore')







def compute_hyperparameters_combination(hyperparameters):
    if type(hyperparameters) is dict:
        hyperparameters_keys=[i for i in hyperparameters.keys()]
        hyperparameters_list=[]
        for i in range(len(hyperparameters)):
            hyperparameters_list.append([j for j in hyperparameters[hyperparameters_keys[i]]])            
        return pd.DataFrame(itertools.product(*hyperparameters_list),columns=hyperparameters_keys)
    elif type(hyperparameters) is pd.core.frame.DataFrame:
        return hyperparameters
    elif type(hyperparameters) is str:
        return pd.read_csv(hyperparameters)
    
    



class regressionGridSearchCV:
        
    def __init__(self,model=None,hyperparameters=None,cv=5):
        self._result_file_name='result.csv'
        if model is not None:
             self._model=model
        else:
            self._model=None  
        if hyperparameters is not None:
            self._hyperparameters_df=compute_hyperparameters_combination(hyperparameters)
            self._result_df=pd.DataFrame(columns=list(i for i in self._hyperparameters_df.keys()) \ 
            +list(['Time','CV_mean_poisson_deviance','CV_mean_absolute_error','CV_mean_absolute_percentage_error','CV_r2_score']))
            self._job_file_name='model_job.csv'
            if not os.path.exists(self._job_file_name):
                self._remaining_job_df=self._hyperparameters_df.copy()
                self._remaining_job_df.to_csv(self._job_file_name,index=False)
            else:
                self._remaining_job_df=pd.read_csv(self._job_file_name)
        else:
            self._hyperparameters_df=None           
                
        self._cv=cv
    
    def _initialize_fit_methods(self):
        self._accuracy_score_test=0
        self._precision_score_test=0
        self._recall_score_test=0
        self._f1_score_test=0
        
        
    def _get_cv_scores(self,ytest,y_test_predicted):
        self._mean_poisson_deviance_test+=mean_poisson_deviance(y_true=ytest,y_pred=y_test_predicted)
        self._mean_absolute_error_test+=mean_absolute_error(y_true=ytest,y_pred=y_test_predicted)
        self._recall_score_test+=mean_absolute_percentage_error(y_true=ytest,y_pred=y_test_predicted)
        self._r2_score_test+=r2_score(y_true=ytest,y_pred=y_test_predicted)
    
    def _get_mean_cv_scores(self):
        self._mean_poisson_deviance_test/=self._cv
        self._mean_absolute_error_test/=self._cv
        self._recall_score_test/=self._cv
        self._r2_score_test/=self._cv
        
    
    def fit(self,X,y):
        if self._model is not None and self._hyperparameters_df is not None:
            self._remaining_job_df=pd.read_csv(self._job_file_name)
            while not self._remaining_job_df.empty:
                start=datetime.now()
                self._initialize_fit_methods()
                self._model.set_params(**(self._remaining_job_df.loc[self._remaining_job_df.index[0]].to_dict()))
                cv_count=1
                rskf = KFold(n_splits=self._cv, n_repeats=1,random_state=0)
                for train_index, test_index in rskf.split(X, y,):
                    xtrain,ytrain,xtest,ytest=X[train_index],y[train_index],X[test_index],y[test_index]
                    self._model.fit(xtrain,ytrain)
                    y_test_predicted=self._model.predict(X=xtest)
                    self._get_cv_scores(ytest,y_test_predicted)
                    print(len(self._remaining_job_df),cv_count,datetime.now(),sep=' : ')
                    cv_count+=1
                self._get_mean_cv_scores()
                temp=pd.concat([self._remaining_job_df.loc[self._remaining_job_df.index[0]],pd.Series([datetime.now().ctime(),self._accuracy_score_test,self._precision_score_test,self._recall_score_test,self._f1_score_test],index=['Time','CV_Accuracy','CV_Precision','CV_Recall','CV_F1Score'])])
                if self._result_df.empty:
                    temp.name=0
                else:
                    temp.name=self._result_df.index[-1]+1
                if not os.path.exists(self._result_file_name):
                    self._result_df=self._result_df.append(temp)                
                    self._result_df.to_csv(self._result_file_name,index=False)
                else:
                    self._result_df=pd.read_csv(self._result_file_name)
                    self._result_df=self._result_df.append(temp)                
                    self._result_df.to_csv(self._result_file_name,index=False)
                self._remaining_job_df.drop(labels=self._remaining_job_df.index[0],inplace=True)
                self._remaining_job_df.to_csv(self._job_file_name,index=False)
                
                
                stop=datetime.now()
                print('Time taken',(stop - start).total_seconds(),sep=' : ')
            else:
                print('Job file is complete, if you want to start new delete:',self._job_file_name)
        else:
            print('Set all Parameters')
    
    
    def get_hyperparameters(self):
        return self._hyperparameters_df
    
    def set_hyperparameters(self,hyperparameters):
        self._hyperparameters_df=compute_hyperparameters_combination(hyperparameters)
        self._result_df=pd.DataFrame(columns=list(i for i in self._hyperparameters_df.keys())+list(['Time','CV_Accuracy','CV_Precision','CV_Recall','CV_F1Score']))
        self._job_file_name='model_job.csv'
        if not os.path.exists(self._job_file_name):            
            self._remaining_job_df=self._hyperparameters_df.copy()
            self._remaining_job_df.to_csv(self._job_file_name,index=False)
        else:
            self._remaining_job_df=pd.read_csv(self._job_file_name)
        return self.get_hyperparameters()
    
    def get_job_file_name(self):
        return self._job_file_name
    
    
    def get_model(self):
        return self._model
    
    def set_model(self,model):
        self._model=model  
        self.get_model()
    
    def get_result_file_name(self):
        return self._result_file_name
    
        
    def get_result(self):
        return self._result_df
    
    def set_result(self,*input_files):
        temp=pd.DataFrame()
        for i in input_files:
            temp=temp.append(pd.read_csv(i))
        self._result_df=temp.copy()
        'CV_mean_poisson_deviance','CV_mean_absolute_error','CV_mean_absolute_percentage_error','CV_r2_score'
    def get_best_score(self,score=None):
        if not self._result_df.empty:
            if score=='poisson':
                return self._result_df['CV_mean_poisson_deviance'].max()
            if score=='mae':
                return self._result_df['CV_mean_absolute_error'].max()
            if score=='mape':
                return self._result_df['CV_mean_absolute_percentage_error'].max()
            if score=='r2':
                return self._result_df['CV_r2_score'].max()
            if score is None:
                print('Pass a scoring metric')
        else:
            print('Set result file for analysis.')
            
    def get_best_hyperparameters(self,score=None,number_of_hyperparameters=1):
        if not self._result_df.empty:
            if score=='poisson':
                return self._result_df.iloc[self._result_df['CV_mean_poisson_deviance'].idxmax(),0:number_of_hyperparameters].to_dict()
            if score=='mae':
                return self._result_df.iloc[self._result_df['CV_mean_absolute_error'].idxmax(),0:number_of_hyperparameters].to_dict()
            if score=='mape':
                return self._result_df.iloc[self._result_df['CV_mean_absolute_percentage_error'].idxmax(),0:number_of_hyperparameters].to_dict()
            if score=='r2':
                return self._result_df.iloc[self._result_df['CV_r2_score'].idxmax(),0:number_of_hyperparameters].to_dict()
            if score is None:
                print('Pass a scoring metric')
        else:
            print('Set result file for getting best hyperparameters.')

    def get_best_model(self,score=None,number_of_hyperparameters=1):
        if self._model is None:
            print('Set a model')
        elif score is None:
            print('Pass a scoring metric')
        else:
            return self._model.set_params(**(self.get_best_hyperparameters(score=score,number_of_hyperparameters=number_of_hyperparameters)))
        