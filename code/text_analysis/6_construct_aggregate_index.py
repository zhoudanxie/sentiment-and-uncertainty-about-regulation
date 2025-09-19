# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import datetime
import os

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
output_folder='data/processed_data'

# %%
# Import data
# Sentiment score data
df=pd.read_csv(f'{directory}/../../data/processed_data/sentiment_scores.csv')

# %%
# Change variable types
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].dt.year
df['Month']=df['StartDate'].dt.month
df['Newspaper']=df['Newspaper'].astype('category')

#%%
# Estimate sentiment indexes

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
# Export indexes
sentimentIndex.to_csv(f'{directory}/../../{output_folder}/aggregate_sentiment_indexes.csv',index=False)
print(f'Aggregate sentiment indexes are saved in the {output_folder} folder.')

#%%
# Estimate news attention index

# %%
# Get monthly count
df_monthly=df.groupby(['Newspaper','Year','Month'],observed=False)['ID'].count().reset_index(name="RegRelevantCount")
df_monthly=df_monthly.sort_values(['Newspaper','Year','Month']).reset_index(drop=True)

#%%
# Total news article counts by publication
df_all=pd.read_excel(f'{directory}/../../data/processed_data/total_article_counts_by_newspaper.xlsx')

# Total article counts by newspaper
df_all=df_all.groupby(['Year','Month','Newspaper']).agg({'Count':'sum'}).reset_index()

#%%
# Merge total article count and reg relevant article count
df_monthly=df_monthly.merge(df_all,on=['Newspaper','Year','Month'],how='left')
df_monthly['year-month']=df_monthly['Year'].astype(int).map(str)+'-'+df_monthly['Month'].astype(int).map(str)

# Function to calculate index
newspapers=df_monthly['Newspaper'].unique()
df_index=df_monthly[['year-month']].drop_duplicates().reset_index(drop=True)

T1_start="1985-1"
T1_end="2009-12"
T2_start="1985-1"
T2_end="2009-12"
def calulate_index(reg_var,total_var):
    df_monthly['X']=df_monthly[reg_var]/df_monthly[total_var]
    # Standardization over T1
    for newspaper in newspapers:
        df_monthly.loc[df_monthly['Newspaper']==newspaper,'variance']=\
            np.var(df_monthly[(df_monthly['Newspaper']==newspaper) &
            (T1_start <= df_monthly['year-month']) & (df_monthly['year-month'] <= T1_end)]['X'])
    df_monthly['Y']=df_monthly['X']/np.sqrt(df_monthly['variance'])
    # Multi-paper index
    for month in df_index['year-month']:
        df_index.loc[df_index['year-month'] == month, 'Z'] = np.mean(
            df_monthly[df_monthly['year-month'] == month]['Y'])
    # Normalization over T2
    M=np.mean(df_index.loc[(T2_start<=df_index['year-month']) & (df_index['year-month']<=T2_end),'Z'])
    new_var=df_index['Z']*(100/M)
    return new_var

# Compute index
df_index['RegRelevance']=calulate_index('RegRelevantCount','Count')

#%%
# Export index
df_index.drop('Z',axis=1).to_csv(f'{directory}/../../{output_folder}/news_attention_index.csv',index=False)
print(f'The index of news attention to regulation is saved in the {output_folder} folder.')


