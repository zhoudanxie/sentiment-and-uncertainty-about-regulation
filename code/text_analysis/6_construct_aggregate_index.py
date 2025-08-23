# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import datetime
import os

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
# directory="text_analysis"

# %%
# Import data
# Sentiment score data
df=pd.read_csv(f'{directory}/../../data/sentiment_scores.csv')

# %%
# Change variable types
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].dt.year
df['Month']=df['StartDate'].dt.month
df['Newspaper']=df['Newspaper'].astype('category')

# %%
# Specify index start date and end date
start_date=datetime.datetime(1985,1,1)
end_date=datetime.datetime(2021,12,31)

df=df[(df['StartDate']>=start_date) & (df['StartDate']<=end_date)].sort_values('StartDate').reset_index(drop=True)

# %%
# Create a DF with all unique years and months
df_ym=df[['Year','Month']].drop_duplicates().reset_index(drop=True).reset_index()
df_ym['YM']=df_ym['index']+1
df_ym=df_ym.drop('index',axis=1)

df=df.merge(df_ym[['Year','Month','YM']],on=['Year','Month'],how='left').sort_values(['Year','Month']).reset_index(drop=True)

# %%
# Define a function to estimate index (suppressing constant)
def estimate_index(var_name):
    FE_OLS=sm.ols(formula=var_name + ' ~ 0+C(YM)+C(Newspaper)',
        data=df).fit()
    # print(FE_OLS.summary())

    FE_estimates=pd.DataFrame()
    FE_estimates[var_name+'Index']=FE_OLS.params[0:max(df_ym['YM'])]
    FE_estimates=FE_estimates.reset_index().rename(columns={'index':'FE'})
    FE_estimates['YM']=FE_estimates['FE'].str.split("[",expand=True)[1].str.split("]",expand=True)[0].astype('int64')
    
    return FE_estimates

# %%
# Uncertainty index
UncertaintyIndex=estimate_index('UncertaintyScore')

# LM index
LMindex=estimate_index('LMscore')

# GI index
GIindex=estimate_index('GIscore')

# LSD index
LSDindex=estimate_index('LSDscore')

# %%
# Merge indexes
sentimentIndex=df_ym.\
        merge(UncertaintyIndex.drop('FE',axis=1),on='YM',how='outer').\
        merge(LMindex.drop('FE',axis=1),on='YM',how='outer').\
        merge(GIindex.drop('FE',axis=1),on='YM',how='outer').\
        merge(LSDindex.drop('FE',axis=1),on='YM',how='outer').\
        sort_values(['Year','Month'])

sentimentIndex=sentimentIndex.\
        rename(columns={'UncertaintyScoreIndex':'UncertaintyIndex','LMscoreIndex':'LMindex',
                        'GIscoreIndex':'GIindex','LSDscoreIndex':'LSDindex'})

# %%
# Export
sentimentIndex.to_csv(f'{directory}/../../data/aggregate_sentiment_indexes.csv',index=False)
print('Aggregate sentiment indexes saved in the /data folder.')



