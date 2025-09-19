# %%
import numpy as np
import pandas as pd
import datetime
import statsmodels.formula.api as sm
import os
import ast

# %%
# Set director
directory=os.path.dirname(os.path.realpath(__file__))

# %% [markdown]
# ## 1. Import article sentiment scores
df=pd.read_csv(f'{directory}/../../data/processed_data/sentiment_scores.csv')

# %%
# Reformat data
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].dt.year
df['Month']=df['StartDate'].dt.month
df['Newspaper']=df['Newspaper'].astype('category')
df['DominantDistinctArea']=df['DominantDistinctArea'].apply(ast.literal_eval)

# Rename column
df=df.rename(columns={'UncertaintyScore':'Uncertaintyscore'})

# %%
# Specify index start date and end date
start_date=datetime.datetime(1985,1,1)
end_date=datetime.datetime(2021,12,31)
end_month=end_date.strftime('%b%Y')

df=df[(df['StartDate']>=start_date) & (df['StartDate']<=end_date)].sort_values('StartDate').reset_index(drop=True)

# %%
# Create year-month dataframe
df_ym=df[['Year','Month']].drop_duplicates().sort_values(['Year','Month']).reset_index(drop=True).reset_index()
df_ym['YM']=df_ym['index']+1
df_ym['YM']=df_ym['YM'].astype('str')
df_ym=df_ym.drop('index',axis=1)
YM_list=df_ym['YM'].tolist()

# Merge year-month dataframe
df=df.merge(df_ym[['Year','Month','YM']],on=['Year','Month'],how='left').sort_values(['Year','Month']).reset_index(drop=True)

# %% [markdown]
# ## 2. Estimate categorical indexes

# %%
# Define a function (suppressing constant) to estimate categorical index
def estimate_categorical_index(score, area_no):
    # Refine data to articles where the dominant area is area_no
    df_area=df[df['DominantDistinctArea'].apply(lambda lst: area_no in lst)]

    # Regression
    FE_OLS=sm.ols(formula=score + ' ~ 0+C(YM)+C(Newspaper)', data=df_area).fit()
    #print(FE_OLS.summary())

    # Clean results
    FE_estimates=pd.DataFrame()
    new_var=score.split('score')[0]+'_DominantDistinctArea'+str(area_no)
    FE_estimates[new_var]=FE_OLS.params[0:len(df_ym)]
    FE_estimates=FE_estimates.reset_index().rename(columns={'index':'FE'})
    FE_estimates['YM']=FE_estimates['FE'].str.split("[",expand=True)[1].str.split("]",expand=True)[0]

    FE_estimates=FE_estimates[FE_estimates['YM'].isin(YM_list)]
    FE_estimates=FE_estimates.drop('FE',axis=1)
    
    return FE_estimates

# %%
# Categorical uncertainty indexes for all areas
area_range=15
CategoricalIndex=df_ym
for area_no in range(1,area_range):
    estimates=estimate_categorical_index('Uncertaintyscore', area_no)
    CategoricalIndex=CategoricalIndex.merge(estimates,on='YM',how='left')

# %%
# Categorical sentiment indexes for all areas
for dict in ['GI','LM','LSD']:
    for area_no in range(1,area_range):
        estimates=estimate_categorical_index(dict+'score', area_no)
        CategoricalIndex=CategoricalIndex.merge(estimates,on='YM',how='left')

# %%
# Export
output_folder='data/processed_data'
CategoricalIndex.to_csv(f'{directory}/../../{output_folder}/categorical_sentiment_indexes.csv',index=False)
print(f'Categorical sentiment indexes are saved in the {output_folder} folder.')