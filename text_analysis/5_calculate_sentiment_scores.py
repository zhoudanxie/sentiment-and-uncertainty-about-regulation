# %%
import pandas as pd
import os
import datetime
import pickle
import re
import time

# %%
# All unique articles
df=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')
print(df.info())

# %%
# Use filtered noun chunk matches to define reg relevance
df.loc[df['NounChunkMatchFiltered']!=0,'RegRelevance']=1
print(df.info())

# %%
# LM uncertainty scores
LMuncertainty=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LMuncertainty.csv')
print(LMuncertainty.info())

# %%
LMuncertainty['ID']=LMuncertainty['ID'].astype('str')
print(LMuncertainty.info())

# %%
# Merge
df2=df.merge(LMuncertainty,on='ID',how='right')
print(df2.info())

# %%
# GI sentiments
GIsentiments=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/GIsentiments.csv')
GIsentiments['ID']=GIsentiments['ID'].astype('str')
print(GIsentiments.info())

# %%
# LSD sentiments
LSDsentiments=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LSDsentiments.csv')
LSDsentiments['ID']=LSDsentiments['ID'].astype('str')
print(LSDsentiments.info())

# %%
# LM sentiments
LMsentiments=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LMsentiments.csv')
LMsentiments['ID']=LMsentiments['ID'].astype('str')
print(LMsentiments.info())

# %%
# Merge
df3=df2.merge(GIsentiments,on='ID',how='left').\
    merge(LMsentiments,on='ID',how='left').\
    merge(LSDsentiments,on='ID',how='left')
print(df3.info())

# %%
# Calculate scores
df3['UncertaintyScore']=df3['UncertaintyCount']/df3['TotalWordCount']*100
for dic in ['GI','LSD','LM']:
    df3[dic+'score']=(df3[dic+'posCount']-df3[dic+'negCount'])/df3['TotalWordCount']*100
print(df3.info())

# %%
print(df3[df3['TotalWordCount']==0][['ID','RegSentsExpand','NounChunkMatchFiltered']])

# %%
# Export all data
df3.to_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_ArticleSentimentWordsScores.pkl')

# %%
# Export sentiment scores only
sentimentScores=df3[['ID','StartDate','Newspaper','PubTitle','UncertaintyScore','GIscore','LMscore','LSDscore']]
print(sentimentScores.info())

# %%
sentimentScores.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_ArticleSentimentScores.csv',index=False)

# %%



