import pandas as pd
import os
import re
import numpy as np
from datetime import datetime
import json
from ast import literal_eval

# Plotting Packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

from statsmodels.tsa.stattools import grangercausalitytests

#%%
from wordcloud import WordCloud
from PIL import Image

from sklearn.decomposition import PCA
import scipy.stats

import spacy
nlp = spacy.load("en_core_web_sm")

#%%
# Common variables
colors=['#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F',
        '#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F']

area_range=15   # Number of areas + 1
dict_area={'1': 'consumer safety and health',
    '2': 'national and homeland security',
    '3': 'transportation',
    '4': 'labor and workplace',
    '5': 'environment and natural resources',
    '6': 'energy',
    '7': 'finance and banking',
    '8': 'general business and trade',
    '9': 'agriculture and rural development',
    '10': 'education and culture',
    '11': 'communications',
    '12': 'criminal justice',
    '13': 'housing, urban development, and social security',
    '14': 'international relations'}

# %%
# Set directory
# directory=os.path.dirname(os.path.realpath(__file__))
directory='code/visualization'

# # Create an output directory if it does not exist
output_folder=f'{directory}/../../figures/appendix_figures'
os.makedirs(output_folder, exist_ok=True)


#-----------------------------------------------------------------------------------------------------------------------
#%%------------------------------------------------Noun Chunk Occurrences-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Appendix A: The Most Common Regulatory Noun Chunks in News Articles
filtered_occurences=pd.read_csv('Data/Aggregate Indexes/RegSentsExpand_FilteredNounChunkOccurences_Dec2021.csv')
print(filtered_occurences.info())

dict_occurences=filtered_occurences.iloc[0:100].set_index('Noun Chunks').to_dict()['Occurences']
print(dict_occurences)
print(len(dict_occurences))

with open('Figures/Manuscript Figures - June 2025/AppendixA.txt', 'w') as convert_file:
    convert_file.write(json.dumps(dict_occurences))

#%% Appendix N: The Most Common Area-specific Regulatory Noun Chunks by Regulatory Area (Total Occurrences)
# Notes: This is only the noun chunks used for classification of each area, so a noun chunk that is counted in the articles
# classified into Area X would not be counted in the articles classified into Area Y.
# For example, "federal reserve" is counted in articles classified into finance and banking regulation, but it is not
# counted in articles for other areas even if they also mention "federal reserve".
filtered_occurences_area=pd.read_csv('Data/Categorical Indexes/RegArea_AreaSpecificNounChunkOccurences_TotalOccurrenceFilter_Dec2021.csv')
print(filtered_occurences_area.info())

# Top 30 noun chunks
areas=list(dict_area.keys())
top_nounchunks=[]
unique_nounchunk_no=[]

for i in range(14):
    var='Occurences_dda'+str(i+1)
    temp_area=filtered_occurences_area.sort_values(var,ascending=False).reset_index()
    df_top30=temp_area.iloc[0:30]
    df_top30[var]=df_top30[var].astype(int)
    top30=df_top30.set_index('Noun Chunks').to_dict()[var]
    top_nounchunks.append(top30)
    nounchunk_no=temp_area[temp_area[var].notnull()]['Noun Chunks'].nunique()
    unique_nounchunk_no.append(nounchunk_no)

df_top_nounchunks=pd.DataFrame()
df_top_nounchunks['Area No']=list(range(1,15))
df_top_nounchunks['Top 30 Noun Chunks and Occurences']=top_nounchunks
df_top_nounchunks['Number of Unique Noun Chunks']=unique_nounchunk_no
df_top_nounchunks['Area Name']=""
for i in range(0,len(df_top_nounchunks)):
    df_top_nounchunks['Area Name'][i]=dict_area[str(df_top_nounchunks['Area No'][i])]

df_top_nounchunks[['Area No','Area Name','Number of Unique Noun Chunks','Top 30 Noun Chunks and Occurences']].\
    to_excel('Figures/Manuscript Figures - June 2025/AppendixN.xlsx',index=False)

#%% Word Clouds of Area-specific Regulatory Noun Chunks by Regulatory Area (Total Occurrences)
filtered_occurences_area=pd.read_csv('Data/Categorical Indexes/RegArea_AreaSpecificNounChunkOccurences_TotalOccurrenceFilter_Dec2021.csv')
print(filtered_occurences_area.info())

font_path="Resources/coolvetica/coolvetica rg.otf"

# Get noun chunk counts
def get_nounchunks(area):
    var='Occurences_dda'+str(area)
    temp_area=filtered_occurences_area.sort_values(var,ascending=False).reset_index()
    nc_dict=temp_area[temp_area[var].notnull()].set_index('Noun Chunks').to_dict()[var]
    return nc_dict

# Function to create a word cloud
def create_wordcloud(area):
    mask=np.array(Image.open('Resources/wordcloud_mask.png'))
    wordcloud=WordCloud(background_color="white",max_words=50,random_state=123,
                        width=500,height=400,font_path=font_path,mask=mask)
    wordcloud.fit_words(get_nounchunks(area))

    return wordcloud

def get_nc_count(area):
    var='Occurences_dda'+str(area)
    n=len(filtered_occurences_area[filtered_occurences_area[var].notnull()])
    return n

# Plot all areas
fig, axes = plt.subplots(5, 3, figsize=(45,40), sharex=False, sharey=False)

for i,ax in enumerate(axes.flatten()):
    if i<len(dict_area):
        ax.imshow(create_wordcloud(i+1),interpolation="bilinear")
        title=dict_area[str(i+1)]
        ax.set_title(dict_area[str(i+1)]+"\n"+"(n = "+str(get_nc_count(i+1))+")",fontsize=46,fontweight='bold')
        ax.axis('off')

fig.delaxes(axes[4,2])
plt.subplots_adjust(hspace=0.3)

plt.tight_layout()
plt.savefig('Figures/Manuscript Figures - June 2025/AppendixN.jpg', bbox_inches='tight')
plt.close()