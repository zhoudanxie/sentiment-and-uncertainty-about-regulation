# %%
import pandas as pd
import os
import datetime
import pickle
import re
import time
from collections import Counter
import numpy as np
import random

# %%
import nltk
nltk.data.path
from nltk.tokenize import sent_tokenize, word_tokenize

# %%
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ## 3.1 Match regulatory noun chunks

# %%
# Define a text preprocessor (lemmatizer)
def my_preprocessor(text):
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_punct | token.is_space]
    texts_out=" ".join(lemmas)
    return texts_out

# %%
# Use the dictionary of regulatory noun chunks
df_nounchunks=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/DictionaryOfRegulatoryNounChunks.csv')
print(df_nounchunks.info())

# %%
nounchunks=df_nounchunks['noun_chunks'].tolist()
print(len(nounchunks),nounchunks[0:20])

# %%
# Import expanded reg sentences
df_regSentsExpand=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/allRegSentsExpand.pkl')
print(df_regSentsExpand.info())

# %%
# Convert reg sentences to list
regSentsExpand=df_regSentsExpand['RegSentsExpand'].tolist()
print(len(regSentsExpand), regSentsExpand[0])

# %%
print(regSentsExpand[4031])
print(my_preprocessor(regSentsExpand[4031]))

# %%
# Preprocess all expanded reg sentences
regSentsExpand_lemmatized=[my_preprocessor(sent) for sent in regSentsExpand]

# %%
print(len(regSentsExpand_lemmatized),regSentsExpand[0], regSentsExpand_lemmatized[0])

# %%
# Compile a new re pattern with regulatory noun chunks
pattern=re.compile(r"\b"+r"\b|\b".join(map(re.escape, nounchunks))+r"\b",re.IGNORECASE)

# %%
# Match noun chunks in all expanded reg sentences
start_time = time.time()

nounchunk_match=[]
nounchunk_match_words=[]
for sent in regSentsExpand_lemmatized:
    match_words=[]
    match=0
    find=pattern.findall(sent)
    if len(find)>0:
        match_words=find
        match=len(find)
    nounchunk_match.append(match)
    nounchunk_match_words.append(match_words)
    
print("--- %s seconds ---" % (time.time() - start_time))

# %%
print(len(nounchunk_match), len(nounchunk_match_words))
print(nounchunk_match_words[-1], nounchunk_match[-1])

# %%
# Export results: matched noun chunks in expanded reg sentences
df_regSentsExpand['NounChunkMatchFiltered']=nounchunk_match
df_regSentsExpand['NounChunkMatchWordsFiltered']=nounchunk_match_words

# %%
print(df_regSentsExpand.head())

# %%
print('# of reg relevant artciles:',df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0]['ID'].nunique())

# %%
print(df_regSentsExpand[df_regSentsExpand['RegSentExpandLength']==0]['NounChunkMatchFiltered'].value_counts())

# %%
# Examine some examples
for i in range(0,100):
    if df_regSentsExpand['NounChunkMatchFiltered'][i]>0:
        print(df_regSentsExpand['RegSentsExpand'][i],df_regSentsExpand['NounChunkMatchWordsFiltered'][i])

# %%
df_regSentsExpand.to_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')

# %% [markdown]
# ## 3.2 Get monthly relevant article counts

# %%
df_regSentsExpand=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')

# %%
# Reg relevant articles
df_reg=df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0].reset_index(drop=True)
print(df_reg.info())

# %%
# Get monthly count
df_monthly=df_reg.groupby(['Newspaper','Year','Month'])['ID'].count().reset_index(name="RegRelevantCount")
df_monthly=df_monthly.sort_values(['Newspaper','Year','Month']).reset_index(drop=True)
print(df_monthly)

# %%
df_monthly.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_MonthlyArticleCount_Dec2021.csv',index=False)

# %% [markdown]
# ### 3.2.1. Original coverage

# %%
print(df_reg['PubTitle'].value_counts())

# %%
df_reg=df_reg[(df_reg['PubTitle']!='Chicago Tribune (Online)')
                & (df_reg['PubTitle']!='Los Angeles Times (Online)')
                & (df_reg['PubTitle']!='New York Times (Online)')
                & (df_reg['PubTitle'] != 'The Washington Post (Online)')].reset_index(drop=True)
print(df_reg['PubTitle'].value_counts())

# %%
# Get monthly count
df_monthly2=df_reg.groupby(['Newspaper','Year','Month'])['ID'].count().reset_index(name="RegRelevantCount")
df_monthly2=df_monthly2.sort_values(['Newspaper','Year','Month']).reset_index(drop=True)
print(df_monthly2)

# %%
df_monthly2.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegRelevant_MonthlyArticleCount_Dec2021_OriginalCoverage.csv',index=False)

# %% [markdown]
# ## 3.3 Noun Chunk Occurences acorss Regulation-related Articles

# %%
df_regSentsExpand=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')
print(df_regSentsExpand.info())

# %%
# Regulation-related articles
df_reg=df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0].reset_index(drop=True)
print(df_reg.info())

# %%
# Append all filtered noun chunks
allMatchWords=[]
for nc_list in df_reg['NounChunkMatchWordsFiltered']:
    nc_list_lower=[nc.lower() for nc in nc_list]    #Convert to lower case
    allMatchWords=allMatchWords+nc_list_lower
print(len(allMatchWords), allMatchWords[0])

# %%
# Count each noun chunk
allMatchWordsCount=Counter(allMatchWords)
print(allMatchWordsCount)

# %%
# Convert to dataframe
df_MatchWords = pd.DataFrame(allMatchWordsCount.items(),columns = ['Noun Chunks','Occurences'])
print(df_MatchWords.info())

# %%
df_MatchWords=df_MatchWords.sort_values('Occurences',ascending=False).reset_index(drop=True)
print(df_MatchWords.head(10))

# %%
# Export noun chunk occurences
df_MatchWords.to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_FilteredNounChunkOccurences.csv',index=False)

# %%



