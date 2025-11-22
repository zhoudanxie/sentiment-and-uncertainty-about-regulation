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
#%%----------------------------Aggregate Regulatory Sentiment and Uncertainty Indexes-----------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Sentiment indexes
monthlyIndex=pd.read_csv(f'{directory}/../../data/processed_data/aggregate_sentiment_indexes.csv')
print(monthlyIndex.info())

monthlyIndex['Year-Month']=monthlyIndex['Year'].map(str)+'-'+monthlyIndex['Month'].map(str)
monthlyIndex['date']=monthlyIndex['Year-Month'].astype('datetime64[ns]').dt.date

#-----------------------------------------------------------------------------------------------------------------------
#%%-----------------------------------Compare Regulatory Indexes with Other Indexes-------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Economic sentiment (Shapiro et al.)
newssent=pd.read_excel(f'{directory}/../../data/raw_data/Shapiro_news_sentiment_data.xlsx', sheet_name='Data')
print(newssent.info())

newssent['Year']=newssent['date'].dt.year
newssent['Month']=newssent['date'].dt.month
newssent_monthly=newssent[['Year','Month','News Sentiment']].groupby(['Year','Month']).agg('mean').reset_index()

monthlyIndex=monthlyIndex.merge(newssent_monthly,on=['Year','Month'],how='left')

# Original BBD EPU
BBD_EPU=pd.read_excel(f'{directory}/../../data/raw_data/EPU_BBD.xlsx')
print(BBD_EPU.info())
BBD_EPU=BBD_EPU[BBD_EPU['News_Based_Policy_Uncert_Index'].notnull()]

BBD_EPU['Year']=BBD_EPU['Year'].astype(int)
BBD_EPU['Month']=BBD_EPU['Month'].astype(int)
monthlyIndex=monthlyIndex.merge(BBD_EPU,on=['Year','Month'],how='left')

# Original BBD REPU
BBD_REPU=pd.read_excel(f'{directory}/../../data/raw_data/Categorical_EPU_Data_BBD.xlsx')
print(BBD_REPU.info())
BBD_REPU=BBD_REPU[BBD_REPU['8. Regulation'].notnull()]
BBD_REPU['Year']=BBD_REPU['Date'].astype('datetime64[ns]').dt.year
BBD_REPU['Month']=BBD_REPU['Date'].astype('datetime64[ns]').dt.month
monthlyIndex=monthlyIndex.merge(BBD_REPU[['Year','Month','8. Regulation']],on=['Year','Month'],how='left')

# Standardize to mean=0 and variance=1
def standardize(data_series):
    standardized_data=(data_series-np.mean(data_series))/np.std(data_series)
    return standardized_data

monthlyIndex['EconomicSentiment_standardized']=standardize(monthlyIndex['News Sentiment'])
monthlyIndex['BBD_EPU_standardized']=standardize(monthlyIndex['News_Based_Policy_Uncert_Index'])
monthlyIndex['BBD_REPU_standardized']=standardize(monthlyIndex['8. Regulation'])

print(monthlyIndex.info())

#%% Appendix E1: Compare Regulatory Sentiment Index and Economic Sentiment Index
x=monthlyIndex['date']
y1=monthlyIndex['LMIndex_standardized']
y2=monthlyIndex['EconomicSentiment_standardized']

# Correlation
print('Correlation between Regulatory Sentiment Index and Economic Sentiment Index:',
      scipy.stats.pearsonr(y1,y2))

fig, ax = plt.subplots(1, figsize=(16,10))
ax.plot(x,y1,color=colors[0],linewidth=1.5,label='Regulatory Sentiment Index')
ax.plot(x,y2,color=colors[1],linewidth=1.5,label='Economic Sentiment Index of Shapiro et al. (2020)')

# events
ax.axvspan(datetime(1991,1,1),datetime(1991,2,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1991,1,1), -3.3, 'Gulf\nWar I', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1993,9,1),datetime(1993,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1993,9,1), 3.6, 'Clinton\nHealth Care Plan', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), -2.2, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2003,3,1),datetime(2003,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2003,3,1), -2.8, 'Gulf\nWar II', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2006,11,1),datetime(2006,12,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2006,11,1), 2.5, 'Bush\nMidterm Election', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), -3, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 2, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,4,1), 2.5, 'Deepwater Horizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 3.5, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2011,8,1),datetime(2011,9,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2011,8,1), -3.3, 'Debt\nCeiling\nDispute', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2012,7,1),datetime(2012,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2012,7,1), -4, 'Libor\nScandal', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),3.8, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), -4.4, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Index',fontsize=16)
ax.set_yticks(np.arange(-6,7,2))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Legend
fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=14)
fig.subplots_adjust(bottom=0.15)

plt.savefig(f'{output_folder}/AppendixE1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix E2: Compare Regulatory Uncertainty Index and Economic Policy Uncertainty Index
x=monthlyIndex['date']
y1=monthlyIndex['UncertaintyIndex_standardized']
y4=monthlyIndex['BBD_EPU_standardized']

# Correlation
print('Correlation between Regulatory Uncertainty Index and Economic Policy Uncertainty Index:',
      scipy.stats.pearsonr(y1,y4))

fig, ax = plt.subplots(1, figsize=(16,10))
ax.plot(x,y1,color=colors[0],label='Regulatory Uncertainty Index')
ax.plot(x,y4,color=colors[1],label='Economic Policy Uncertainty Index of Baker et al. (2016)')

# events
ax.axvspan(datetime(1987,10,1),datetime(1987,11,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1987,10,1), 2.2, 'Black\nMonday', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1991,1,1),datetime(1991,2,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1991,1,1), 2, 'Gulf\nWar I', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1994,5,1),datetime(1994,6,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1994,5,1), 2, 'GAO Proposal\nfor Derivative\nRegulations', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), 3, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2003,3,1),datetime(2003,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2003,3,1), 2.2, 'Gulf\nWar II', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), 3, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 4.2, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,4,1), 4.5, 'Deepwater Horizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 5.2, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2011,8,1),datetime(2011,9,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2011,8,1), 3.1, 'Debt\nCeiling\nDispute', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),3.8, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), 7, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Index',fontsize=16)
ax.set_yticks(np.arange(-4,9,2))
ax.set_ylim(bottom=-4)
ax.grid(color='#d3d3d3', which='major', axis='y')

fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
fig.subplots_adjust(bottom=0.15)

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig(f'{output_folder}/AppendixE2.jpg', bbox_inches='tight')
plt.close()

#%% Appendix E3: Compare Regulatory Uncertainty Index and Regulatory EPU Index
x=monthlyIndex['date']
y1=monthlyIndex['UncertaintyIndex_standardized']
y3=monthlyIndex['BBD_REPU_standardized']

# Correlation
print('Correlation between Regulatory Uncertainty Index and Regulatory EPU Index:',
      scipy.stats.pearsonr(y1,y3))

fig, ax = plt.subplots(1, figsize=(16,10))
ax.plot(x,y1,color=colors[0],label='Regulatory Uncertainty Index')
ax.plot(x,y3,color=colors[1],label='Regulatory EPU Index of Baker et al. (2016)')

# events
ax.axvspan(datetime(1987,10,1),datetime(1987,11,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1987,10,1), 2.2, 'Black\nMonday', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1991,1,1),datetime(1991,2,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1991,1,1), 4.5, 'Gulf\nWar I', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1994,5,1),datetime(1994,6,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1994,2,1), 2, 'GAO Proposal\nfor Derivative\nRegulations', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), 2, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2003,3,1),datetime(2003,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2003,3,1), 1.5, 'Gulf\nWar II', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), 4.2, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 5.2, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,4,1), 5.6, 'Deepwater Horizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 6.5, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2011,8,1),datetime(2011,9,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2011,9,1), 3.5, 'Debt\nCeiling\nDispute', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),3.6, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), 4.7, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Index',fontsize=16)
ax.set_yticks(np.arange(-4,9,2))
ax.set_ylim(bottom=-4)
ax.grid(color='#d3d3d3', which='major', axis='y')

fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
fig.subplots_adjust(bottom=0.15)

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig(f'{output_folder}/AppendixE3.jpg', bbox_inches='tight')
plt.close()

#%% Appendix E4: Granger causality
# H0: the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1.

print('Does regulatory sentiment Granger causes economic sentiment?')
gc_res = grangercausalitytests(monthlyIndex[['EconomicSentiment_standardized','LMIndex_standardized']], 4)

print('Does economic sentiment Granger causes regulatory sentiment?')
gc_res = grangercausalitytests(monthlyIndex[['LMIndex_standardized','EconomicSentiment_standardized']], 4)

print('Does regulatory uncertainty Granger causes economic policy uncertainty?')
gc_res = grangercausalitytests(monthlyIndex[['BBD_EPU_standardized','UncertaintyIndex_standardized']], 4)

print('Does economic policy uncertainty Granger causes regulatory uncertainty?')
gc_res = grangercausalitytests(monthlyIndex[['UncertaintyIndex_standardized','BBD_EPU_standardized']], 4)

print('Does regulatory uncertainty Granger causes regulatory EPU?')
gc_res = grangercausalitytests(monthlyIndex[['BBD_REPU_standardized','UncertaintyIndex_standardized']], 4)

print('Does regulatory EPU Granger causes regulatory uncertainty?')
gc_res = grangercausalitytests(monthlyIndex[['UncertaintyIndex_standardized','BBD_REPU_standardized']], 4)

#%% Appendix F: Stationarity Tests for the Regulatory Sentiment and Uncertainty Indexes
# Augmented Dickey-Fuller test (H0: non-stationary)
def adf_test(var):
    x=monthlyIndex[var]
    adf = ADF(x,trend="ct")
    print('Results of Augmented Dickey-Fuller Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(adf.stat))
    print('p-value:','{0:0.6f}'.format(adf.pvalue))
    print('Lags:',adf.lags)
    #print(adf.summary())

# Phillips-Perron test (H0: non-stationary)
def pp_test(var):
    x=monthlyIndex[var]
    pp = PhillipsPerron(x,trend="ct")
    print('Results of Phillips-Perron Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(pp.stat))
    print('p-value:','{0:0.6f}'.format(pp.pvalue))
    print('Lags:',pp.lags)

# KPSS test (H0: stationary)
def kpss_test(var):
    x=monthlyIndex[var]
    kpss = KPSS(x,trend="ct")
    print('Results of KPSS Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(kpss.stat))
    print('p-value:','{0:0.6f}'.format(kpss.pvalue))
    print('Lags:',kpss.lags)

# Tests for all indexes
for var in ['UncertaintyIndex','LMIndex','GIIndex','LSDIndex','SentimentPC1_standardized']:
    adf_test(var)
    pp_test(var)
    kpss_test(var)
