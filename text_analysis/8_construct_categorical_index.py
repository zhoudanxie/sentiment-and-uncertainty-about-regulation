# %%
import numpy as np
import pandas as pd
import datetime
import statsmodels.formula.api as sm

# %% [markdown]
# ## 1. Import reg-relevant article sentiment scores

# %%
df=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_ArticleSentimentScores.csv')
print(df.info())

# %%
# Reformat data
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].dt.year
df['Month']=df['StartDate'].dt.month
df['Newspaper']=df['Newspaper'].astype('category')
#print(df.info())

# %%
# Specify index start date and end date
start_date=datetime.datetime(1985,1,1)
end_date=datetime.datetime(2021,12,31)
end_month=end_date.strftime('%b%Y')

df=df[(df['StartDate']>=start_date) & (df['StartDate']<=end_date)].sort_values('StartDate').reset_index(drop=True)
print(df[['StartDate','Year','Month']])

# %%
# Create year-month dataframe
df_ym=df[['Year','Month']].drop_duplicates().sort_values(['Year','Month']).reset_index(drop=True).reset_index()
df_ym['YM']=df_ym['index']+1
df_ym['YM']=df_ym['YM'].astype('str')
df_ym=df_ym.drop('index',axis=1)
print(df_ym)

# %%
# Merge year-month dataframe
df=df.merge(df_ym[['Year','Month','YM']],on=['Year','Month'],how='left').sort_values(['Year','Month']).reset_index(drop=True)
print(df.info())

# %% [markdown]
# ## 2. Estimate categorical indexes

# %%
df=df.rename(columns={'UncertaintyScore':'Uncertaintyscore'})

# %%
YM_list=df_ym['YM'].tolist()
#print(YM_list)

# %%
# Define a function (suppressing constant) to estimate categorical index
def estimate_categorical_index(score, area):
    df_area=df[df[area]==1].reset_index(drop=True)
    FE_OLS=sm.ols(formula=score + ' ~ 0+C(YM)+C(Newspaper)', data=df_area).fit()
    #print(FE_OLS.summary())

    FE_estimates=pd.DataFrame()
    new_var=score.split('score')[0]+'_'+area
    FE_estimates[new_var]=FE_OLS.params[0:len(df_ym)]
    FE_estimates=FE_estimates.reset_index().rename(columns={'index':'FE'})
    FE_estimates['YM']=FE_estimates['FE'].str.split("[",expand=True)[1].str.split("]",expand=True)[0]
    
    for value in FE_estimates['YM']:
        if value not in YM_list:
            FE_estimates=FE_estimates[FE_estimates['YM']!=value]
    FE_estimates=FE_estimates.drop('FE',axis=1)
    
    return FE_estimates

# %%
# List of columns for all areas
area_range=15
area_list=[]
for i in range(1,area_range):
    var='DominantDistinctArea'+str(i)
    area_list.append(var)

# %%
# Define another function (with constant) to estimate categorical index
def estimate_categorical_index_constant(score, area):
    df_area=df[df[area]==1].reset_index(drop=True)
    FE_OLS=sm.ols(formula=score + ' ~ C(YM)+C(Newspaper)', data=df_area).fit()
    #print(FE_OLS.summary())

    FE_estimates=pd.DataFrame()
    new_var=score.split('score')[0]+'_'+area
    FE_estimates['coef']=FE_OLS.params[0:len(df_ym)]
    FE_estimates=FE_estimates.reset_index().rename(columns={'index':'FE'})
    
    for value in FE_estimates['FE']:
        if ('YM' not in value) & ('Intercept' not in value):
            FE_estimates=FE_estimates[FE_estimates['YM']!=value]
    
    intercept=FE_estimates[FE_estimates['FE']=='Intercept']['coef'].values
    FE_estimates.loc[FE_estimates['FE']!='Intercept',new_var]=FE_estimates.loc[FE_estimates['FE']!='Intercept','coef']+intercept
    FE_estimates.loc[FE_estimates['FE']=='Intercept',new_var]=FE_estimates.loc[FE_estimates['FE']=='Intercept','coef']
    FE_estimates.loc[FE_estimates['FE']=='Intercept','FE']='C(YM)[T.1]'
    FE_estimates=FE_estimates[['FE',new_var]].reset_index(drop=True)
    FE_estimates['YM']=FE_estimates['FE'].str.split("T.",expand=True)[1].str.split("]",expand=True)[0]
    FE_estimates=FE_estimates.drop('FE',axis=1)
    
    return FE_estimates

# %%
# Categorical Uncertainty Index
CategoricalUncertaintyIndex=df_ym
for area in area_list:
    try:
        estimates=estimate_categorical_index('Uncertaintyscore', area)
        CategoricalUncertaintyIndex=CategoricalUncertaintyIndex.merge(estimates,on='YM',how='left')
    except:
        print("Failed:",area)
        estimates=estimate_categorical_index_constant('Uncertaintyscore', area)
        CategoricalUncertaintyIndex=CategoricalUncertaintyIndex.merge(estimates,on='YM',how='left')

# %%
print(CategoricalUncertaintyIndex.info())

# %%
print(CategoricalUncertaintyIndex[['Year','Month','Uncertainty_DominantDistinctArea1',
                                   'Uncertainty_DominantDistinctArea2']].head())

# %%
CategoricalUncertaintyIndex.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_MonthlyUncertaintyIndex_'+str(end_month)+'.csv',index=False)

# %%
# Categorical sentiment indexes
for dict in ['GI','LM','LSD']:
    CategoricalSentimentIndex=df_ym
    for area in area_list:
        try:
            estimates=estimate_categorical_index(dict+'score', area)
            CategoricalSentimentIndex=CategoricalSentimentIndex.merge(estimates,on='YM',how='left')
        except:
            print("Failed:",dict+":"+area)
            estimates=estimate_categorical_index_constant(dict+'score', area)
            CategoricalSentimentIndex=CategoricalSentimentIndex.merge(estimates,on='YM',how='left')        
    CategoricalSentimentIndex.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_Monthly'+dict+'Index_'+str(end_month)+'.csv',index=False)

# %%



