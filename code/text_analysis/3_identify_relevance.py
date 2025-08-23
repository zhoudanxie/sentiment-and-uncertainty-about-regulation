# %%
import pandas as pd
import os
import re
import time
from collections import Counter
import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
# directory="text_analysis"

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
df_nounchunks=pd.read_csv(f'{directory}/supplementary_data/DictionaryOfRegulatoryNounChunks.csv')

nounchunks=df_nounchunks['noun_chunks'].tolist()
print('Number of regulatory noun chunks:', len(nounchunks),nounchunks[0:20])

# %%
# Import expanded reg sentences
df_regSentsExpand=pd.read_pickle(f'{directory}/sample_data/sample_output/reg_sections.pkl')
# print(df_regSentsExpand.info())

# %%
# Convert reg sentences to list
regSentsExpand=df_regSentsExpand['RegSection'].tolist()

# %%
# Preprocess all expanded reg sentences
regSentsExpand_lemmatized=[my_preprocessor(sent) for sent in regSentsExpand]

# %%
# Compile a new re pattern with regulatory noun chunks
pattern=re.compile(r"\b"+r"\b|\b".join(map(re.escape, nounchunks))+r"\b",re.IGNORECASE)

# %%
# Match noun chunks in all regulatory sections
print("Searching regulatory noun chunks in all regulatory sections...")
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
# Export results: matched noun chunks in expanded reg sentences
df_regSentsExpand['NounChunkMatchFiltered']=nounchunk_match
df_regSentsExpand['NounChunkMatchWordsFiltered']=nounchunk_match_words

# %%
print('# of reg relevant articles:',df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0]['ID'].nunique())

# %%
df_regSentsExpand.to_pickle(f'{directory}/sample_data/sample_output/reg_sections.pkl')

# %% [markdown]
# ## 3.2 Get monthly relevant article counts

# %%
# Reg relevant articles
df_reg=df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0].reset_index(drop=True)

# %%
# Get monthly count
df_monthly=df_reg.groupby(['Newspaper','Year','Month'])['ID'].count().reset_index(name="RegRelevantCount")
df_monthly=df_monthly.sort_values(['Newspaper','Year','Month']).reset_index(drop=True)

# %%
df_monthly.to_csv(f'{directory}/sample_data/sample_output/RegRelevant_MonthlyArticleCount.csv',index=False)

# %% [markdown]
# ## 3.3 Noun Chunk Occurences acorss Regulation-related Articles

# %%
# Append all filtered noun chunks
allMatchWords=[]
for nc_list in df_reg['NounChunkMatchWordsFiltered']:
    nc_list_lower=[nc.lower() for nc in nc_list]    #Convert to lower case
    allMatchWords=allMatchWords+nc_list_lower

# %%
# Count each noun chunk
allMatchWordsCount=Counter(allMatchWords)

# %%
# Convert to dataframe
df_MatchWords = pd.DataFrame(allMatchWordsCount.items(),columns = ['Noun Chunks','Occurences'])
df_MatchWords=df_MatchWords.sort_values('Occurences',ascending=False).reset_index(drop=True)

# %%
# Export noun chunk occurences
df_MatchWords.to_csv(f'{directory}/sample_data/sample_output/RegSections_FilteredNounChunkOccurrences.csv',index=False)