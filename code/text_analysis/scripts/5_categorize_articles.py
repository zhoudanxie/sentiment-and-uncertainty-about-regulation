# %%
import pandas as pd
import os
import pickle
import re
from collections import Counter
import numpy as np
from ast import literal_eval

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))

# Set output directory
output_folder=f'{directory}/../../data/raw_data/text_sample_data/sample_output'

# %% [markdown]
# ## 1. Import Regulatory Sections and Noun Chunks with Areas

# %%
# Noun chunks with areas
nounchunks_area=pd.read_csv(f'{directory}/../../data/processed_data/dictionary_of_regulatory_noun_chunks.csv')

# Convert to dictionary
nounchunks_area=nounchunks_area[nounchunks_area['area_no']>0].set_index('noun_chunks')
nounchunks_area_dict=nounchunks_area.to_dict()['area']

# %%
# Regulatory sections with matched noun chunks
df_regSentsExpand=pd.read_pickle(f'{output_folder}/sentiment_scores.pkl')

# Refine to reg relevant articles
df_regSentsExpandRelevant=df_regSentsExpand[df_regSentsExpand['RegRelevance']==1].reset_index(drop=True)

# %% [markdown]
# ## 2. Link Expanded Reg Sentences to Areas

# %% [markdown]
# ### Approach 4: dominant distinct area (dda): use the dominant areas from area-specific noun chunks (approach adopted in paper)

# %%
# Get all areas associated with area-specific noun chunks
distinct_areas=[]
for nounchunks in df_regSentsExpandRelevant['NounChunkMatchWordsFiltered']:
    area_list=[]
    for nc in nounchunks:
        if nc in nounchunks_area_dict:
            area=sorted(literal_eval(nounchunks_area_dict[nc]))
            if len(area)==1:
                area_list=area_list+area
    distinct_areas.append(area_list)

df_regSentsExpandRelevant['AllDistinctAreas']=distinct_areas

# %%
# Get the dominant area(s)
distinct_area_counts=[]
dominant_distinct_areas=[]
for area_list in df_regSentsExpandRelevant['AllDistinctAreas']:
    area_count=Counter(area_list).most_common()
    dominant_area=[j for j in Counter(area_list).keys() if area_list.count(j)==max(Counter(area_list).values())]
    distinct_area_counts.append(area_count)
    dominant_distinct_areas.append(dominant_area)

df_regSentsExpandRelevant['DistinctAreaCount']=distinct_area_counts
df_regSentsExpandRelevant['DominantDistinctArea']=dominant_distinct_areas

# %% [markdown]
# ### Create dummies for areas

# %%
# Separate dominant areas to different columns
# area_range=15    # Number of areas + 1
# for i in range(1,area_range):
#     var='DominantDistinctArea'+str(i)
#     df_regSentsExpandRelevant[var]=0
#     for j in range(0, len(df_regSentsExpandRelevant)):
#         if i in df_regSentsExpandRelevant['DominantDistinctArea'][j]:
#             df_regSentsExpandRelevant[var][j]=1

area_range=15    # Number of areas + 1
for i in range(1,area_range):
    var='DominantDistinctArea'+str(i)
    var_values=[]
    for area_list in df_regSentsExpandRelevant['DominantDistinctArea']:
        if i in area_list:
            var_values.append(1)
        else:
            var_values.append(0)
    df_regSentsExpandRelevant[var]=var_values

# %%
# Save data
df_regSentsExpandRelevant.to_pickle(f'{output_folder}/sentiment_scores.pkl')

# %% [markdown]
# ## 4. Monthly article counts by area
# Area count for each article
dda_list=df_regSentsExpandRelevant['DominantDistinctArea'].tolist()
area_counts=[]
for area in dda_list:
    count=len(area)
    area_counts.append(count)
df_regSentsExpandRelevant['DominantDistinctAreaCount']=area_counts

# %%
# Total article count
print("# of articles with an area classification:",
      len(df_regSentsExpandRelevant[df_regSentsExpandRelevant['DominantDistinctAreaCount']>0]))

# %%
# List of columns for different areas
col_list=[]
for i in range(1,area_range):
    var='DominantDistinctArea'+str(i)
    col_list.append(var)

# %%
# Aggregate monthly article counts
monthlyAreaCount=df_regSentsExpandRelevant[['Newspaper','Year','Month']+col_list].\
                groupby(['Newspaper','Year','Month']).agg('sum').reset_index()

# %%
# Save data
monthlyAreaCount.to_csv(f'{output_folder}/article_counts_by_newspaper_and_area.csv',index=False)

# %% [markdown]
# ## 5. Filtered Noun Chunk Occurences by Area
# Filtered noun chunk occurences across regulation-related articles
df_nounchunk_occurrences=pd.read_csv(f'{output_folder}/noun_chunk_occurrences.csv')

# %%
# Filtered noun chunks across regulation-related articles by area
for i in range(1,15):
    allMatchWords=[]
    for nc_list in df_regSentsExpandRelevant[df_regSentsExpandRelevant['DominantDistinctArea'+str(i)]==1]['NounChunkMatchWordsFiltered']:
        nc_list_lower = [nc.lower() for nc in nc_list]
        allMatchWords=allMatchWords+nc_list_lower
    allMatchWordsCount=Counter(allMatchWords)
    var_name='OccurrencesArea'+str(i)
    df_MatchWords = pd.DataFrame(allMatchWordsCount.items(),columns = ['Noun Chunks',var_name])
    df_nounchunk_occurrences=df_nounchunk_occurrences.merge(df_MatchWords,on='Noun Chunks',how='outer')

# %%
# Save data
df_nounchunk_occurrences.to_csv(f'{output_folder}/noun_chunk_occurrences.csv',index=False)




