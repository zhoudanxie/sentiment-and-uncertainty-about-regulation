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
#%%-------------------------Categorical Regulatory Sentiment and Uncertainty Indexes------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#%% Appendix O: Frequencies of Articles By Regulatory Area
areas=pd.DataFrame()
areas['area_id']=list(dict_area.keys())
areas['area_title']=list(dict_area.values())

# Determine which human checking approach to use
robust='TotalOccurrence'
# Determine area classification name
area='DominantDistinctArea'

articles_area=pd.read_csv(f'{directory}/../../data/processed_data/RegArea_MonthlyArticleCountByNewspaper_'+robust+'Filter_Dec2021.csv')
print(articles_area.info())

col_list=[]
for i in range(1,15):
    new=area+str(i)
    col_list.append(new)

monthlyArticleCountByArea=articles_area[['Year','Month']+col_list].groupby(['Year','Month']).agg('sum')
print((monthlyArticleCountByArea.info()))

totalArticleCountByArea=monthlyArticleCountByArea.sum(axis=0).reset_index().\
    rename(columns={'index':'area',0:'article_count'})

# Merge in area titles
totalArticleCountByArea['area_id']=totalArticleCountByArea['area'].str.split('Area',expand=True)[1]
totalArticleCountByArea=totalArticleCountByArea.merge(areas,on='area_id',how='left')

# Plot article counts
totalArticleCountByArea=totalArticleCountByArea.sort_values('article_count').reset_index(drop=True)

fig, ax = plt.subplots(1, figsize=(17, 10))
fig.subplots_adjust(left=0.3)

ax.barh(totalArticleCountByArea['area_title'], totalArticleCountByArea['article_count'], height=.6, color=colors[0])

ax.tick_params(axis='x', labelsize=14, color='#d3d3d3')
ax.tick_params(axis='y', labelsize=16, color='#d3d3d3')
ax.set_xlabel('Number of Articles (N=305,367)',fontsize=14)     # N = number of articles with at least one reg area
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xticks(np.arange(0,max(totalArticleCountByArea['article_count'])+10000,20000))

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig(f'{output_folder}/AppendixQ.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
# Import categorical indexes
sentiment_area=pd.read_csv(f'{directory}/../../data/processed_data/categorical_sentiment_indexes.csv')
sentiment_area['Year']=sentiment_area['Year'].astype('int64')
sentiment_area['Month']=sentiment_area['Month'].astype('int64')
sentiment_area['year-month']=sentiment_area['Year'].map(str)+'-'+sentiment_area['Month'].map(str)
sentiment_area['date']=sentiment_area['year-month'].astype('datetime64[ns]').dt.date

#%% Appendix P1: Monthly Sentiment Index By Regulatory Policy Area
x=sentiment_area['date']

fig, axes = plt.subplots(5, 3, figsize=(50,45), sharex=False, sharey=False)

years = mdates.YearLocator(3)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

for i, ax in enumerate(axes.flatten()):
    if i+1<15:
        y1=sentiment_area['LM_'+area+str(i+1)]
        y2=sentiment_area['LM_'+area+str(i+1)].rolling(window=12,center=True).mean()
        ax.plot(x, y1, color=colors[0])
        ax.plot(x, y2, color=colors[7], linestyle='dashed', linewidth=2.5)
        ax.set_title(dict_area[str(i+1)],fontsize=42)
    else:
        break

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    datemin = np.datetime64(sentiment_area['date'].iloc[0], 'Y')
    datemax = np.datetime64(sentiment_area['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    ax.grid(False)

    ax.tick_params(axis='both', which='major', labelsize=24, color='gray')
    ax.tick_params(axis='both', which='minor', color='gray')

    # borders
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False, rotation=30)
fig.delaxes(axes[4,2])
fig.text(0.09, 0.55, 'Regulatory Sentiment Index', va='center', rotation='vertical', fontsize=42)
plt.subplots_adjust(wspace=0.1,hspace=0.2)

plt.savefig(f'{output_folder}/AppendixR1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix P2: Monthly Uncertainty Index By Regulatory Policy Area
x=sentiment_area['date']

fig, axes = plt.subplots(5, 3, figsize=(50,45), sharex=False, sharey=False)

years = mdates.YearLocator(3)   # every X year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

for i, ax in enumerate(axes.flatten()):
    if i+1<15:
        y1=sentiment_area['Uncertainty_'+area+str(i+1)]
        y2=sentiment_area['Uncertainty_'+area+str(i+1)].rolling(window=12,center=True).mean()
        ax.plot(x, y1, color=colors[0])
        ax.plot(x, y2, color=colors[7], linestyle='dashed', linewidth=2.5)
        ax.set_title(dict_area[str(i+1)],fontsize=42)
    else:
        break

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    datemin = np.datetime64(sentiment_area['date'].iloc[0], 'M')
    datemax = np.datetime64(sentiment_area['date'].iloc[-1], 'M') + np.timedelta64(1, 'M')
    ax.set_xlim(datemin, datemax)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    ax.grid(False)

    ax.tick_params(axis='both', which='major', labelsize=24, color='gray')
    ax.tick_params(axis='both', which='minor', color='gray')

    # borders
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False, rotation=30)
fig.delaxes(axes[4,2])
fig.text(0.09, 0.55, 'Regulatory Uncertainty Index', va='center', rotation='vertical', fontsize=42)
plt.subplots_adjust(wspace=0.1,hspace=0.2)

plt.savefig(f'{output_folder}/AppendixR2.jpg', bbox_inches='tight')
plt.close()
