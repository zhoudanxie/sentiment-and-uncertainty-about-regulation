# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import datetime

# %%
# Updated sentiment score data
df_new=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_ArticleSentimentWordsScores.pkl')
print(df_new.info())

print("Updated Start Dates:")
print(df_new.sort_values('StartDate')[['StartDate']])

# %%
# # Append previous and updated data
# df=df_old.append(df_new).drop_duplicates('ID').sort_values(['StartDate','Newspaper'])
# print(df.info())
# print(df[['StartDate']])

# %%
# Reg relevant articles
df=df_new[df_new['RegRelevance']==1].sort_values(['StartDate','PubTitle'])
print(df.info())

# %%
# Change variable types
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].dt.year
df['Month']=df['StartDate'].dt.month
df['Newspaper']=df['Newspaper'].astype('category')
print(df.info())

# %%
# Check on publications considered
print(df['PubTitle'].value_counts())

# %%
# Specify index start date and end date
start_date=datetime.datetime(1985,1,1)
end_date=datetime.datetime(2021,12,31)
end_month=end_date.strftime('%b%Y')

df=df[(df['StartDate']>=start_date) & (df['StartDate']<=end_date)].sort_values('StartDate').reset_index(drop=True)
print(df[['StartDate','Year','Month']])

# %%
# Create a DF with all unique years and months
df_ym=df[['Year','Month']].drop_duplicates().reset_index(drop=True).reset_index()
df_ym['YM']=df_ym['index']+1
df_ym=df_ym.drop('index',axis=1)
print(df_ym)

# %%
df=df.merge(df_ym[['Year','Month','YM']],on=['Year','Month'],how='left').sort_values(['Year','Month']).reset_index(drop=True)
print(df.info())

# %%
print(df[['ID','UncertaintyScore','GIscore','LMscore','LSDscore']].head())

# %%
print(df['Newspaper'].value_counts())

# %%
# Revised function to estimate index (suppressing constant)
def estimate_index(var_name):
    FE_OLS=sm.ols(formula=var_name + ' ~ 0+C(YM)+C(Newspaper)',
        data=df).fit()
    print(FE_OLS.summary())

    FE_estimates=pd.DataFrame()
    FE_estimates[var_name+'Index']=FE_OLS.params[0:max(df_ym['YM'])]
    FE_estimates=FE_estimates.reset_index().rename(columns={'index':'FE'})
    FE_estimates['YM']=FE_estimates['FE'].str.split("[",expand=True)[1].str.split("]",expand=True)[0].astype('int64')
    
    return FE_estimates

# %%
# Uncertainty index
UncertaintyIndex=estimate_index('UncertaintyScore')

# %%
# LM index
LMindex=estimate_index('LMscore')

# %%
# GI index
GIindex=estimate_index('GIscore')

# %%
# LSD index
LSDindex=estimate_index('LSDscore')

# %%
# Merge indexes
sentimentIndex=df_ym.merge(UncertaintyIndex,on='YM',how='outer').\
        merge(LMindex,on='YM',how='outer').\
        merge(GIindex,on='YM',how='outer').\
        merge(LSDindex,on='YM',how='outer').\
        sort_values(['Year','Month'])
print(sentimentIndex.info())

# %%
sentimentIndex=sentimentIndex.drop(['FE_x','FE_y'],axis=1).\
        rename(columns={'UncertaintyScoreIndex':'UncertaintyIndex','LMscoreIndex':'LMindex',
                        'GIscoreIndex':'GIindex','LSDscoreIndex':'LSDindex'})

# %%
print(sentimentIndex.head())

# %%
print(sentimentIndex.info())
print(sentimentIndex.tail())

# %%
# Export
sentimentIndex.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_MonthlySentimentIndex_'+str(end_month)+'.csv',index=False)

# %%



