# %%
import pandas as pd
import os
import datetime
import pickle
import re
import time
from collections import Counter
import numpy as np
from ast import literal_eval

# %%
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %% [markdown]
# ## 1. Import Regulatory Sections and Noun Chunks with Areas

# %%
# Noun chunks with areas
nounchunks_area=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/DictionaryOfRegulatoryNounChunks.csv')
print(nounchunks_area.info())

# %%
nounchunks_area.head()

# %%
# Convert to dictionary
nounchunks_area=nounchunks_area[nounchunks_area['area_no']>0].set_index('noun_chunks')
nounchunks_area_dict=nounchunks_area.to_dict()['area']
print(len(nounchunks_area_dict))

# %%
# Expanded reg sentences with matched noun chunks
df_regSentsExpand=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')
print(df_regSentsExpand.info())

# %%
# Refine to reg relevant articles
df_regSentsExpandRelevant=df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0].reset_index(drop=True)
print(df_regSentsExpandRelevant.info())

# %% [markdown]
# ## 2. Link Expanded Reg Sentences to Areas

# %% [markdown]
# ### Approach 4: dominant distinct area (dda): use the dominant areas from area-specific noun chunks (approach adopted in paper)

# %%
# Get all areas associated with area-specific noun chunks
df_regSentsExpandRelevant['AllDistinctAreas']=''
for i in range(0, len(df_regSentsExpandRelevant)):
    nounchunks=df_regSentsExpandRelevant['NounChunkMatchWordsFiltered'][i]
    area_list=[]
    for nc in nounchunks:
        if nc in nounchunks_area_dict:
            area=sorted(literal_eval(nounchunks_area_dict[nc]))
            if len(area)==1:
                area_list=area_list+area
    df_regSentsExpandRelevant['AllDistinctAreas'][i]=area_list

# %%
# Get the dominant area(s)
df_regSentsExpandRelevant['DistinctAreaCount']=''
df_regSentsExpandRelevant['DominantDistinctArea']=''
for i in range(0, len(df_regSentsExpandRelevant)):
    area_list=df_regSentsExpandRelevant['AllDistinctAreas'][i]
    area_count=Counter(area_list).most_common()
    dominant_area=[j for j in Counter(area_list).keys() if area_list.count(j)==max(Counter(area_list).values())]
    df_regSentsExpandRelevant['DistinctAreaCount'][i]=area_count
    df_regSentsExpandRelevant['DominantDistinctArea'][i]=dominant_area
print(df_regSentsExpandRelevant.info())

# %%
print(df_regSentsExpandRelevant[['DistinctAreaCount','DominantDistinctArea','AllDistinctAreas']].head())

# %% [markdown]
# ### Create dummies for areas

# %%
area_range=15
for i in range(1,area_range):
    var='DominantDistinctArea'+str(i)
    df_regSentsExpandRelevant[var]=0
    for j in range(0, len(df_regSentsExpandRelevant)):
        if i in df_regSentsExpandRelevant['DominantDistinctArea'][j]:
            df_regSentsExpandRelevant[var][j]=1
print(df_regSentsExpandRelevant.info())

# %% [markdown]
# ## 3. Merge with sentiment scores

# %%
# Merge with sentiment scores
sentimentScores=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_ArticleSentimentScores.csv')
print(sentimentScores.info())

# %%
# Merge
sentimentScores['ID']=sentimentScores['ID'].astype('str')
df=df_regSentsExpandRelevant.merge(sentimentScores[['ID','UncertaintyScore','GIscore','LMscore','LSDscore']],on='ID',how='left')
print(df.info())

# %%
df.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_ArticleSentimentScores.csv',index=False)

# %% [markdown]
# ## 4. Monthly article counts by area

# %%
df=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_ArticleSentimentScores.csv',
               converters={'DominantDistinctArea': pd.eval})
print(df.info())

# %%
# Area count for each article
dda_list=df['DominantDistinctArea'].tolist()
area_counts=[]
for area in dda_list:
    count=len(area)
    area_counts.append(count)
df['DominantDistinctAreaCount']=area_counts

# %%
print(df[['DominantDistinctArea','DominantDistinctAreaCount']].tail())

# %%
# Total article count
print("# of articles with an area classification:",len(df[df['DominantDistinctAreaCount']>0]))

# %%
# List of columns for different approaches
area_range=15    # Number of areas + 1
col_list=[]
for i in range(1,area_range):
    var='DominantDistinctArea'+str(i)
    col_list.append(var)
print(col_list)

# %%
# Aggregate monthly article counts
monthlyAreaCount=df[['Newspaper','Year','Month']+col_list].groupby(['Newspaper','Year','Month']).agg('sum').reset_index()
print(monthlyAreaCount.info())

# %%
monthlyAreaCount.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_MonthlyArticleCountByNewspaper.csv',index=False)

# %% [markdown]
# ## 5. Filtered Noun Chunk Occurences by Area

# %%
# Reg relevant articles
df=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_ArticleSentimentScores.csv')
print(df.info())

# %%
# Filtered noun chunk occurences across regulation-related articles
df_nounchunk_occurences=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_FilteredNounChunkOccurences.csv')
print(df_nounchunk_occurences.info())

# %%
# An example
print(df[df['DominantDistinctArea1']==1]['NounChunkMatchWordsFiltered'])

# %%
print(literal_eval(df[df['DominantDistinctArea1']==1]['NounChunkMatchWordsFiltered'][15].lower()))

# %%
# Filtered noun chunks across regulation-related articles by area
for i in range(1,15):
    allMatchWords=[]
    for list in df[df['DominantDistinctArea'+str(i)]==1]['NounChunkMatchWordsFiltered']:
        allMatchWords=allMatchWords+literal_eval(list.lower())
    allMatchWordsCount=Counter(allMatchWords)
    var_name='Occurences_dda'+str(i)
    df_MatchWords = pd.DataFrame(allMatchWordsCount.items(),columns = ['Noun Chunks',var_name])
    df_nounchunk_occurences=df_nounchunk_occurences.merge(df_MatchWords,on='Noun Chunks',how='outer')
print(df_nounchunk_occurences.head())

# %%
df_nounchunk_occurences.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegArea_FilteredNounChunkOccurences.csv',index=False)

# %%



