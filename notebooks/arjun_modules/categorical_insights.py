def categorical_insights(data,catcol,target,plot=False,verbose=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    import statsmodels.api as sm
    dependent_col=[]
    for i in catcol:
        crosstab_df=pd.crosstab(data[i],data[target],margins=True)
        crosstab_array=sm.stats.Table.from_data(data[[i,target]]).table_orig.to_numpy()
        p_value=chi2_contingency(observed=crosstab_array)[1]
        if verbose is True:
            if p_value>0.05:
                print(i,'and',target,'are INDEPENDENT of each other with p-value=',p_value)
            else:
                print(i,'and',target,'are _DEPENDENT of each other with p-value=',p_value)
        if plot is True:
            crosstab_df['p_of_outcome']=crosstab_df[crosstab_df.columns[1]]/crosstab_df[crosstab_df.columns[2]]
            sns.barplot(x=crosstab_df.index,y='p_of_outcome',data=crosstab_df) 
            plt.show()
        if p_value<0.05:
            dependent_col.append(i)
    return dependent_col