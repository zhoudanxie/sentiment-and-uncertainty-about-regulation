#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------Paper Figures, November 7, 2022 Draft------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------------------
#%%---------------------------------Impulse Responses to Aggregate Shocks (VAR)-----------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Set some common variables
steps=36        # Number of steps for IRF

vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# IRF - LM
lm_irf=pd.read_stata('Analysis/VAR/irf_output/irf_lm.dta')
print(lm_irf.info())

lm_irf=lm_irf[lm_irf['step']<=steps]
lm_irf_baseline=lm_irf[lm_irf['irfname']=='baseline'].reset_index(drop=True)

# IRF - RPU
rpu_irf=pd.read_stata('Analysis/VAR/irf_output/irf_rpu.dta')
print(rpu_irf.info())

rpu_irf=rpu_irf[rpu_irf['step']<=steps]
rpu_irf_baseline=rpu_irf[rpu_irf['irfname']=='baseline'].reset_index(drop=True)

#%% Appendix I1: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
# Define a function to plot
def ax_plot(y1,y2,y3,y4,y5):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8',linewidth=0.1)
    ax.plot(x, y3, color='#C8C8C8',linewidth=0.1)
    ax.plot(x, y4, color='#E8E8E8',linewidth=0.1)
    ax.plot(x, y5, color='#E8E8E8',linewidth=0.1)

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
    ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
    ax.axhline(y=0, color='black', linestyle='dotted',linewidth=1.5)

    # title
    ax.set_title(titles[i], fontsize=20)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    #ax.set_yticks(np.arange(-1, 0.6, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabels[i], fontsize=18)
    ax.set_xlabel('Months', fontsize=16)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

# Plot
x=lm_irf_baseline['step']

fig, axes = plt.subplots(2, 2, figsize=(18,12), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    y1 = lm_irf_baseline['oirflm'+vars[i]]
    y2 = lm_irf_baseline['l'+vars[i]+'95']
    y3 = lm_irf_baseline['h'+vars[i]+'95']
    y4 = lm_irf_baseline['l'+vars[i]]
    y5 = lm_irf_baseline['h'+vars[i]]

    ax_plot(y1,y2,y3,y4,y5)

for i, ax in enumerate(axes[1].flatten()):
    y1 = rpu_irf_baseline['oirfrpu' + vars[i]]
    y2 = rpu_irf_baseline['l' + vars[i] + '95']
    y3 = rpu_irf_baseline['h' + vars[i] + '95']
    y4 = rpu_irf_baseline['l' + vars[i]]
    y5 = rpu_irf_baseline['h' + vars[i]]

    ax_plot(y1,y2,y3,y4,y5)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixI1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix I2: Impulse Responses to a Regulatory Sentiment/Uncertainty Shock (Alternative VAR Specifications)
tests=['baseline', 'reverse', 'timetrend', 'vix', 'nsp', 'bi_output', 'rbi_output', 'bi_emp', 'rbi_emp']
tests_labels=['baseline','reverse','timetrend','vix','no s&p','bivariate','bivariate reverse']

# Define a function to plot
def ax_plot(y1,y2,y3,y4,y5,y6,y7):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D", label=tests_labels[0])
    ax.plot(x, y2, color=colors[0], marker=".", label=tests_labels[1])
    ax.plot(x, y3, color=colors[1], marker="v", label=tests_labels[2])
    ax.plot(x, y4, color=colors[2], marker="8", label=tests_labels[3])
    ax.plot(x, y5, color=colors[3], marker="P", label=tests_labels[4])
    ax.plot(x, y6, color=colors[4], marker="+", label=tests_labels[5])
    ax.plot(x, y7, color=colors[5], marker=">", label=tests_labels[6])

    # title
    ax.set_title(titles[i], fontsize=20)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    ax.set_yticks(np.arange(-0.6,0.4,0.2))
    start, end = ax.get_ylim()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabels[i], fontsize=16)
    ax.set_xlabel('Months', fontsize=16)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

x=lm_irf_baseline['step']

fig, axes = plt.subplots(2, 2, figsize=(18,13), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    y1 = lm_irf[lm_irf['irfname']==tests[0]]['oirflm'+vars[i]]
    y2 = lm_irf[lm_irf['irfname']==tests[1]]['oirflm'+vars[i]]
    y3 = lm_irf[lm_irf['irfname']==tests[2]]['oirflm'+vars[i]]
    y4 = lm_irf[lm_irf['irfname']==tests[3]]['oirflm'+vars[i]]
    y5 = lm_irf[lm_irf['irfname']==tests[4]]['oirflm'+vars[i]]
    if i==0:
            y6 = lm_irf[lm_irf['irfname']==tests[5]]['oirflm'+vars[i]]
            y7 = lm_irf[lm_irf['irfname']==tests[6]]['oirflm'+vars[i]]
    if i==1:
            y6 = lm_irf[lm_irf['irfname']==tests[7]]['oirflm'+vars[i]]
            y7 = lm_irf[lm_irf['irfname']==tests[8]]['oirflm'+vars[i]]

    ax_plot(y1, y2, y3, y4, y5, y6, y7)

for i, ax in enumerate(axes[1].flatten()):
    y1 = rpu_irf[rpu_irf['irfname'] == tests[0]]['oirfrpu' + vars[i]]
    y2 = rpu_irf[rpu_irf['irfname'] == tests[1]]['oirfrpu' + vars[i]]
    y3 = rpu_irf[rpu_irf['irfname'] == tests[2]]['oirfrpu' + vars[i]]
    y4 = rpu_irf[rpu_irf['irfname'] == tests[3]]['oirfrpu' + vars[i]]
    y5 = rpu_irf[rpu_irf['irfname'] == tests[4]]['oirfrpu' + vars[i]]
    if i == 0:
        y6 = rpu_irf[rpu_irf['irfname'] == tests[5]]['oirfrpu' + vars[i]]
        y7 = rpu_irf[rpu_irf['irfname'] == tests[6]]['oirfrpu' + vars[i]]
    if i == 1:
        y6 = rpu_irf[rpu_irf['irfname'] == tests[7]]['oirfrpu' + vars[i]]
        y7 = rpu_irf[rpu_irf['irfname'] == tests[8]]['oirfrpu' + vars[i]]

    ax_plot(y1, y2, y3, y4, y5, y6, y7)

# legend
axes[0][0].legend(loc=(0.5, -0.36), ncol=4, fontsize=16)
axes[1][0].legend(loc=(0.5, -0.36), ncol=4, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixI2.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------------------Control for other sentiment/uncertainty----------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Define a function for subplots
def ax_plot(y1, y2, y3, y4, y5, ymin=-1, ymax=0.4):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8')
    ax.plot(x, y3, color='#C8C8C8')
    ax.plot(x, y4, color='#E8E8E8')
    ax.plot(x, y5, color='#E8E8E8')

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
    ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
    ax.axhline(y=0, color='black',linestyle='dotted',linewidth=2.5)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    ax.set_yticks(np.arange(ymin, ymax, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabels[i], fontsize=16)
    ax.set_xlabel('Months', fontsize=16)

    # title
    ax.set_title(titles[i], fontsize=20)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

#%% Appendix J1: Impulse Responses to a Regulatory Sentiment Shock (Controlling for Economic Sentiment and Policy Uncertainty)
# Import IRF output
controls=['mich','newssent','vix','epu']
irf_gdp = pd.DataFrame()
irf_gdp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/lm_lgdp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lgdp_' + c, 'u90': 'llgdp_' + c, 'd90': 'hlgdp_' + c,
                 'u95': 'llgdp95_' + c, 'd95': 'hlgdp95_' + c})
    irf_gdp = irf_gdp.merge(new, on='step').reset_index(drop=True)

irf_emp = pd.DataFrame()
irf_emp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/lm_lemp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lemp_' + c, 'u90': 'llemp_' + c, 'd90': 'hlemp_' + c,
                 'u95': 'llemp95_' + c, 'd95': 'hlemp95_' + c})
    irf_emp = irf_emp.merge(new, on='step').reset_index(drop=True)

irf_output=irf_gdp.merge(irf_emp, on='step').reset_index(drop=True)

# Plot
x=irf_gdp['step']

fig, axes = plt.subplots(4, 2, figsize=(18,16), sharex=False, sharey=False)

for c in range(4):
    for i, ax in enumerate(axes[c].flatten()):
        y1 = irf_output[vars[i]+'_'+controls[c]]
        y2 = irf_output['l'+vars[i]+'95_'+controls[c]]
        y3 = irf_output['h'+vars[i]+'95_'+controls[c]]
        y4 = irf_output['l'+vars[i]+'_'+controls[c]]
        y5 = irf_output['h'+vars[i]+'_'+controls[c]]
        ax_plot(y1, y2, y3, y4, y5)

fig.text(0.5, 0.9, '(a) Controlling for Michigan Consumer Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.69, '(b) Controlling for General Economic Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.48, '(c) Controlling for VIX', ha='center', fontsize=24)
fig.text(0.5, 0.27, '(d) Controlling for Economic Policy Uncertainty', ha='center', fontsize=24)
plt.subplots_adjust(hspace=0.55)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixJ1.jpg', bbox_inches='tight')

#%% Appendix J2: Impulse Responses to a Regulatory Uncertainty Shock (Controlling for Economic Sentiment and Policy Uncertainty)
# Import IRF output
controls=['mich','newssent','vix','epu']
irf_gdp = pd.DataFrame()
irf_gdp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/rpu_lgdp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lgdp_' + c, 'u90': 'llgdp_' + c, 'd90': 'hlgdp_' + c,
                 'u95': 'llgdp95_' + c, 'd95': 'hlgdp95_' + c})
    irf_gdp = irf_gdp.merge(new, on='step').reset_index(drop=True)

irf_emp = pd.DataFrame()
irf_emp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/rpu_lemp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lemp_' + c, 'u90': 'llemp_' + c, 'd90': 'hlemp_' + c,
                 'u95': 'llemp95_' + c, 'd95': 'hlemp95_' + c})
    irf_emp = irf_emp.merge(new, on='step').reset_index(drop=True)

irf_output=irf_gdp.merge(irf_emp, on='step').reset_index(drop=True)

# Plot
x=irf_gdp['step']

fig, axes = plt.subplots(4, 2, figsize=(18,16), sharex=False, sharey=False)

for c in range(4):
    for i, ax in enumerate(axes[c].flatten()):
        y1 = irf_output[vars[i]+'_'+controls[c]]
        y2 = irf_output['l'+vars[i]+'95_'+controls[c]]
        y3 = irf_output['h'+vars[i]+'95_'+controls[c]]
        y4 = irf_output['l'+vars[i]+'_'+controls[c]]
        y5 = irf_output['h'+vars[i]+'_'+controls[c]]
        ax_plot(y1, y2, y3, y4, y5, -0.8, 0.4)

fig.text(0.5, 0.9, '(a) Controlling for Michigan Consumer Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.69, '(b) Controlling for General Economic Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.48, '(c) Controlling for VIX', ha='center', fontsize=24)
fig.text(0.5, 0.27, '(d) Controlling for Economic Policy Uncertainty', ha='center', fontsize=24)
plt.subplots_adjust(hspace=0.55)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixJ2.jpg', bbox_inches='tight')

#-----------------------------------------------------------------------------------------------------------------------
#%%-------------------------Categorical Regulatory Sentiment and Uncertainty Indexes------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
areas=pd.DataFrame()
areas['area_id']=list(dict_area.keys())
areas['area_title']=list(dict_area.values())

# Determine which human checking approach to use
robust='TotalOccurrence'
# Determine area classification name
area='DominantDistinctArea'

# Data
uncertaintyIndexArea=pd.read_csv('Data/Categorical Indexes/RegArea_MonthlyUncertaintyIndex_'+robust+'Filter_Dec2021.csv')
print(uncertaintyIndexArea.info())
GIIndexArea=pd.read_csv('Data/Categorical Indexes/RegArea_MonthlyGIIndex_'+robust+'Filter_Dec2021.csv')
LMIndexArea=pd.read_csv('Data/Categorical Indexes/RegArea_MonthlyLMIndex_'+robust+'Filter_Dec2021.csv')
LSDIndexArea=pd.read_csv('Data/Categorical Indexes/RegArea_MonthlyLSDIndex_'+robust+'Filter_Dec2021.csv')

sentiment_area=uncertaintyIndexArea.merge(GIIndexArea.drop('YM',axis=1),on=['Year','Month'],how='outer').\
            merge(LMIndexArea.drop('YM',axis=1),on=['Year','Month'],how='outer').\
            merge(LSDIndexArea.drop('YM',axis=1),on=['Year','Month'],how='outer')

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

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixP1.jpg', bbox_inches='tight')
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

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixP2.jpg', bbox_inches='tight')
plt.close()

#%% Appendix O: Frequencies of Articles By Regulatory Area
articles_area=pd.read_csv('Data/Categorical Indexes/RegArea_MonthlyArticleCountByNewspaper_'+robust+'Filter_Dec2021.csv')
print(articles_area.info())

area='DominantDistinctArea'
col_list=[]
for i in range(1,15):
    new=area+str(i)
    col_list.append(new)
print(col_list)

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

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixO.jpg', bbox_inches='tight')
plt.close()

#%% Check finance and banking regulation
df_finance=sentiment_area[['Year','Month','LM_DominantDistinctArea7','Uncertainty_DominantDistinctArea7']]
print(df_finance.sort_values('LM_DominantDistinctArea7').head(20))
print(df_finance.sort_values('Uncertainty_DominantDistinctArea7',ascending=False).head(20))

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------Impulse Responses to Categorical Shocks (Local Projections)------------------------------
#-----------------------------------------------------------------------------------------------------------------------
areas=pd.DataFrame()
areas['area_id']=list(dict_area.keys())
areas['area_title']=list(dict_area.values())

# Determine which human checking approach to use
robust='TotalOccurrence'
# Determine area classification name
area='DominantDistinctArea'

# Set number of steps for IRF
steps=12
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# IRF - LM shocks
irf_lm_area=pd.DataFrame()
irf_lm_area['step']=range(0,steps+1)
for i in range(1, area_range):
    new=pd.read_stata('Analysis/Local Projections/output/lm_dda'+str(i)+'_lgdp_'+robust.lower()+'.dta')
    new=new.rename(columns={'Years':'step','b':'lgdp_'+str(i),'u90':'llgdp90_'+str(i),'d90':'hlgdp90_'+str(i),
                            'u95':'llgdp95_'+str(i),'d95':'hlgdp95_'+str(i)})
    irf_lm_area=irf_lm_area.merge(new,on='step',how='outer')

    new=pd.read_stata('Analysis/Local Projections/output/lm_dda'+str(i)+'_lemp_'+robust.lower()+'.dta')
    new=new.rename(columns={'Years':'step','b':'lemp_'+str(i),'u90':'llemp90_'+str(i),'d90':'hlemp90_'+str(i),
                            'u95':'llemp95_'+str(i),'d95':'hlemp95_'+str(i)})
    irf_lm_area=irf_lm_area.merge(new,on='step',how='outer')

# IRF - RPU shocks
irf_rpu_area=pd.DataFrame()
irf_rpu_area['step']=range(0,steps+1)
for i in range(1, area_range):
    new=pd.read_stata('Analysis/Local Projections/output/rpu_dda'+str(i)+'_lgdp_'+robust.lower()+'.dta')
    new=new.rename(columns={'Years':'step','b':'lgdp_'+str(i),'u90':'llgdp90_'+str(i),'d90':'hlgdp90_'+str(i),
                            'u95':'llgdp95_'+str(i),'d95':'hlgdp95_'+str(i)})
    irf_rpu_area=irf_rpu_area.merge(new,on='step',how='outer')

    new=pd.read_stata('Analysis/Local Projections/output/rpu_dda'+str(i)+'_lemp_'+robust.lower()+'.dta')
    new=new.rename(columns={'Years':'step','b':'lemp_'+str(i),'u90':'llemp90_'+str(i),'d90':'hlemp90_'+str(i),
                            'u95':'llemp95_'+str(i),'d95':'hlemp95_'+str(i)})
    irf_rpu_area=irf_rpu_area.merge(new,on='step',how='outer')

#%% Appendix Q1: Industrial Production Responses to a Negative Sentiment Shock By Regulatory Area
x=irf_lm_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_lm_area['lgdp_' + str(i + 1)]
                y2 = irf_lm_area['llgdp95_'+str(i + 1)]
                y3 = irf_lm_area['hlgdp95_'+str(i + 1)]
                y4 = irf_lm_area['llgdp90_'+str(i + 1)]
                y5 = irf_lm_area['hlgdp90_'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,3))
        ax.set_yticks(np.arange(-0.8,0.6,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[0], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix Q2: Employment Responses to a Negative Sentiment Shock By Regulatory Area
x=irf_lm_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_lm_area['lemp_' + str(i + 1)]
                y2 = irf_lm_area['llemp95_'+str(i + 1)]
                y3 = irf_lm_area['hlemp95_'+str(i + 1)]
                y4 = irf_lm_area['llemp90_'+str(i + 1)]
                y5 = irf_lm_area['hlemp90_'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,3))
        ax.set_yticks(np.arange(-0.4,0.5,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[1], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ2.jpg', bbox_inches='tight')
plt.close()

#%% Appendix Q3: Industrial Production Responses to an Uncertainty Shock By Regulatory Area
x=irf_rpu_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_rpu_area['lgdp_' + str(i + 1)]
                y2 = irf_rpu_area['llgdp95_'+str(i + 1)]
                y3 = irf_rpu_area['hlgdp95_'+str(i + 1)]
                y4 = irf_rpu_area['llgdp90_'+str(i + 1)]
                y5 = irf_rpu_area['hlgdp90_'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,3))
        ax.set_yticks(np.arange(-0.4,0.6,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[0], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ3.jpg', bbox_inches='tight')
plt.close()

#%% Appendix Q4: Employment Responses to an Uncertainty Shock By Regulatory Area
x=irf_rpu_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_rpu_area['lemp_' + str(i + 1)]
                y2 = irf_rpu_area['llemp95_'+str(i + 1)]
                y3 = irf_rpu_area['hlemp95_'+str(i + 1)]
                y4 = irf_rpu_area['llemp90_'+str(i + 1)]
                y5 = irf_rpu_area['hlemp90_'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,3))
        ax.set_yticks(np.arange(-0.2,0.3,0.1))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[1], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ4.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------------------------------Quarterly IRFs-----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
vars=['lm','gi','lsd','pc','rpu']
titles=['Regulatory Sentiment Shock (LM)',
        'Regulatory Sentiment Shock (GI)',
        'Regulatory Sentiment Shock (LSD)',
        'Regulatory Sentiment Shock (PC)',
        'Regulatory Uncertainty Shock']

# Function for plot
def ax_plot(y1,y2,y3,y4,y5,ylabel):

        ax.plot(x,y1,color='black',linewidth=2.5,marker="D")
        ax.plot(x,y2,color='#C8C8C8')
        ax.plot(x,y3,color='#C8C8C8')
        ax.plot(x,y4,color='#E8E8E8')
        ax.plot(x,y5,color='#E8E8E8')

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # title
        ax.set_title(titles[i], fontsize=20)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,2))
        ax.tick_params(axis='both',which='major',labelsize=14)
        ax.margins(x=0.01)

        # axis labels
        ax.set_ylabel(ylabel,fontsize=18)
        ax.set_xlabel('Quarters',fontsize=16)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

#%% Appendix H1: GDP Responses to Regulatory Sentiment and Uncertainty Shocks (Quarterly)
steps=12
irf_gdp=pd.DataFrame()
irf_gdp['step']=range(0,steps+1)
for index in vars:
    new=pd.read_stata('Analysis/Local Projections/robustness/output/'+index+'_lgdp_quarterly.dta')
    new=new.rename(columns={'Years':'step','b':'lgdp_'+index,'u90':'llgdp_'+index,'d90':'hlgdp_'+index,'u95':'llgdp95_'+index,'d95':'hlgdp95_'+index})
    irf_gdp=irf_gdp.merge(new,on='step').reset_index(drop=True)

# Plot GDP responses (all)
ylabel='GDP Response, %'
x=irf_gdp['step']

fig = plt.figure(figsize=(14,17))
iplot = 420
for i in range(5):
        iplot += 1
        if i == 4:
                ax = plt.subplot2grid((4,8), (i//2, 2), colspan=4)
        else:
                ax = fig.add_subplot(iplot)

        y1 = irf_gdp['lgdp_'+vars[i]]
        y2 = irf_gdp['llgdp95_'+vars[i]]
        y3 = irf_gdp['hlgdp95_'+vars[i]]
        y4 = irf_gdp['llgdp_'+vars[i]]
        y5 = irf_gdp['hlgdp_'+vars[i]]

        ax_plot(y1, y2, y3, y4,y5,ylabel)
        ax.set_yticks(np.arange(-1.5,0.6,0.5))

plt.subplots_adjust(hspace=0.4,wspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixH1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix H2: Investment Responses to Regulatory Sentiment and Uncertainty Shocks (Quarterly)
steps=12
irf_investment=pd.DataFrame()
irf_investment['step']=range(0,steps+1)
for index in vars:
    new=pd.read_stata('Analysis/Local Projections/robustness/output/'+index+'_lgross_quarterly.dta')
    new=new.rename(columns={'Years':'step','b':'lgross_'+index,'u90':'llgross_'+index,'d90':'hlgross_'+index,'u95':'llgross95_'+index,'d95':'hlgross95_'+index})
    irf_investment=irf_investment.merge(new,on='step').reset_index(drop=True)

# Plot
ylabel='Investment Response, %'
x=irf_investment['step']

fig = plt.figure(figsize=(14,17))
iplot = 420
for i in range(5):
        iplot += 1
        if i == 4:
                ax = plt.subplot2grid((4,8), (i//2, 2), colspan=4)
        else:
                ax = fig.add_subplot(iplot)

        y1 = irf_investment['lgross_'+vars[i]]
        y2 = irf_investment['llgross95_'+vars[i]]
        y3 = irf_investment['hlgross95_'+vars[i]]
        y4 = irf_investment['llgross_' + vars[i]]
        y5 = irf_investment['hlgross_' + vars[i]]

        ax_plot(y1, y2, y3, y4,y5,ylabel)
        ax.set_yticks(np.arange(-4,5,2))

plt.subplots_adjust(hspace=0.4,wspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixH2.jpg', bbox_inches='tight')
plt.close()

#%% Appendix H3: Employment Responses to Regulatory Sentiment and Uncertainty Shocks (Quarterly)
steps=12
irf_emp=pd.DataFrame()
irf_emp['step']=range(0,steps+1)
for index in vars:
    new=pd.read_stata('Analysis/Local Projections/robustness/output/'+index+'_lemp_quarterly.dta')
    new=new.rename(columns={'Years':'step','b':'lemp_'+index,'u90':'llemp_'+index,'d90':'hlemp_'+index,'u95':'llemp95_'+index,'d95':'hlemp95_'+index})
    irf_emp=irf_emp.merge(new,on='step').reset_index(drop=True)

# Plot employment responses (all)
ylabel='Employment Response, %'
x=irf_emp['step']

fig = plt.figure(figsize=(14,17))
iplot = 420
for i in range(5):
        iplot += 1
        if i == 4:
                ax = plt.subplot2grid((4,8), (i//2, 2), colspan=4)
        else:
                ax = fig.add_subplot(iplot)

        y1 = irf_emp['lemp_'+vars[i]]
        y2 = irf_emp['llemp95_'+vars[i]]
        y3 = irf_emp['hlemp95_'+vars[i]]
        y4 = irf_emp['llemp_'+vars[i]]
        y5 = irf_emp['hlemp_'+vars[i]]

        ax_plot(y1, y2, y3, y4,y5,ylabel)
        ax.set_yticks(np.arange(-1.5,0.6,0.5))

plt.subplots_adjust(hspace=0.4,wspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixH3.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------------------Removing Deregulation Articles-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
monthlyIndexRobust=pd.read_csv('Data/Aggregate Indexes/Robustness Checks/RegRelevant_MonthlySentimentIndex_NoDereg_Dec2021.csv')
print(monthlyIndexRobust.info())
monthlyIndexRobust['Year-Month']=monthlyIndexRobust['Year'].map(str)+'-'+monthlyIndexRobust['Month'].map(str)
monthlyIndexRobust['date']=monthlyIndexRobust['Year-Month'].astype('datetime64[ns]').dt.date

monthlyIndex=pd.read_csv('Data/Aggregate Indexes/RegRelevant_MonthlySentimentIndex_Dec2021.csv')
print(monthlyIndex.info())
monthlyIndex['Year-Month']=monthlyIndex['Year'].map(str)+'-'+monthlyIndex['Month'].map(str)
monthlyIndex['date']=monthlyIndex['Year-Month'].astype('datetime64[ns]').dt.date

#%% Appendix L: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Import IRF output
steps=12
lm_irf = pd.DataFrame()
lm_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata('Analysis/Local Projections/output/lm_'+v+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v, 'u90': 'l'+v, 'd90': 'h' + v,
                 'u95': 'l' + v+'95', 'd95': 'h' + v+'95'})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)
    new = pd.read_stata('Analysis/Local Projections/robustness/output/lm_'+v+'_nodereg.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v+'_nodereg', 'u90': 'l'+v+'_nodereg', 'd90': 'h' + v+'_nodereg',
                 'u95': 'l' + v+'95_nodereg', 'd95': 'h' + v+'95_nodereg'})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)

rpu_irf = pd.DataFrame()
rpu_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata('Analysis/Local Projections/output/rpu_'+v+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v, 'u90': 'l'+v, 'd90': 'h' + v,
                 'u95': 'l' + v+'95', 'd95': 'h' + v+'95'})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)
    new = pd.read_stata('Analysis/Local Projections/robustness/output/rpu_'+v+'_nodereg.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v+'_nodereg', 'u90': 'l'+v+'_nodereg', 'd90': 'h' + v+'_nodereg',
                 'u95': 'l' + v+'95_nodereg', 'd95': 'h' + v+'95_nodereg'})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)

# Define a function for subplots
def ax_plot(y1,y2,y3,y4,y5,y6,legend1="Baseline",legend2="Robustness check"):
        ax.plot(x,y2,color='#dcdcdc')
        ax.plot(x,y3,color='#dcdcdc')
        ax.fill_between(x, y2, y3, facecolor='#dcdcdc')
        ax.plot(x,y1,color='black',linewidth=2.5,marker="D",label=legend1)
        ax.plot(x,y5,color=colors[1],linewidth=2.5)
        ax.plot(x,y6,color=colors[1],linewidth=2.5)
        ax.plot(x,y4,color=colors[1],linewidth=2.5,marker="o",label=legend2)

        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # title
        ax.set_title(titles[i], fontsize=20)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,6))
        ax.set_yticks(np.arange(-1,0.5,0.2))
        ax.tick_params(axis='both',which='major',labelsize=14)
        ax.margins(x=0.01)

        # axis labels
        ax.set_ylabel(ylabels[i],fontsize=18)
        ax.set_xlabel('Months',fontsize=16)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

# Plot
x=lm_irf['step']

fig, axes = plt.subplots(2, 2, figsize=(18,14), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
        y1 = lm_irf[vars[i]]
        y2 = lm_irf['l'+vars[i]+'95']
        y3 = lm_irf['h'+vars[i]+'95']
        y4 = lm_irf[vars[i]+'_nodereg']
        y5 = lm_irf['l'+vars[i]+'95_nodereg']
        y6 = lm_irf['h'+vars[i]+'95_nodereg']

        ax_plot(y1,y2,y3,y4,y5,y6,'Baseline: IRF to a Regulatory Sentiment Shock','Robustness check: removing "deregulat*" articles')

for i, ax in enumerate(axes[1].flatten()):
        y1 = rpu_irf[vars[i]]
        y2 = rpu_irf['l'+vars[i]+'95']
        y3 = rpu_irf['h'+vars[i]+'95']
        y4 = rpu_irf[vars[i]+'_nodereg']
        y5 = rpu_irf['l'+vars[i]+'95_nodereg']
        y6 = rpu_irf['h'+vars[i]+'95_nodereg']

        ax_plot(y1,y2,y3,y4,y5,y6,'Baseline: IRF to a Regulatory Uncertainty Shock','Robustness check: removing "deregulat*" articles')

#legend
axes[0][0].legend(loc=(0.25, -0.28), ncol=2, fontsize=16)
axes[1][0].legend(loc=(0.25, -0.28), ncol=2, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixL.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------Impulse Responses to Aggregate Shocks (Local Projections)----------------------------------
#-----------------------------------------------------------------------------------------------------------------------
lm_lgdp=pd.read_stata('Analysis/Local Projections/output/lm_lgdp.dta')
lm_lemp=pd.read_stata('Analysis/Local Projections/output/lm_lemp.dta')
gi_lgdp=pd.read_stata('Analysis/Local Projections/output/gi_lgdp.dta')
gi_lemp=pd.read_stata('Analysis/Local Projections/output/gi_lemp.dta')
lsd_lgdp=pd.read_stata('Analysis/Local Projections/output/lsd_lgdp.dta')
lsd_lemp=pd.read_stata('Analysis/Local Projections/output/lsd_lemp.dta')
pc_lgdp=pd.read_stata('Analysis/Local Projections/output/pc_lgdp.dta')
pc_lemp=pd.read_stata('Analysis/Local Projections/output/pc_lemp.dta')

rpu_lgdp=pd.read_stata('Analysis/Local Projections/output/rpu_lgdp.dta')
rpu_lemp=pd.read_stata('Analysis/Local Projections/output/rpu_lemp.dta')

# Define a function for subplots
def ax_plot(y1, y2, y3, y4, y5, ymin=-1, ymax=0.4):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8')
    ax.plot(x, y3, color='#C8C8C8')
    ax.plot(x, y4, color='#E8E8E8')
    ax.plot(x, y5, color='#E8E8E8')

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
    ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
    ax.axhline(y=0, color='black',linestyle='dotted',linewidth=2.5)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    ax.set_yticks(np.arange(ymin, ymax, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabels[i], fontsize=18)
    ax.set_xlabel('Months', fontsize=16)

    # title
    ax.set_title(titles[i], fontsize=20)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

#%% Impulse Responses to a Regulatory Sentiment Shock (Alternative Measures)
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Plot
x=lm_lgdp['Years']

fig, axes = plt.subplots(3, 2, figsize=(18,15), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    if i==0:
        y1 = gi_lgdp['b']
        y2 = gi_lgdp['u95']
        y3 = gi_lgdp['d95']
        y4 = gi_lgdp['u90']
        y5 = gi_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5)
    if i==1:
        y1 = gi_lemp['b']
        y2 = gi_lemp['u95']
        y3 = gi_lemp['d95']
        y4 = gi_lemp['u90']
        y5 = gi_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5)

for i, ax in enumerate(axes[1].flatten()):
    if i == 0:
        y1 = lsd_lgdp['b']
        y2 = lsd_lgdp['u95']
        y3 = lsd_lgdp['d95']
        y4 = lsd_lgdp['u90']
        y5 = lsd_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5)
    if i == 1:
        y1 = lsd_lemp['b']
        y2 = lsd_lemp['u95']
        y3 = lsd_lemp['d95']
        y4 = lsd_lemp['u90']
        y5 = lsd_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5)

for i, ax in enumerate(axes[2].flatten()):
    if i == 0:
        y1 = pc_lgdp['b']
        y2 = pc_lgdp['u95']
        y3 = pc_lgdp['d95']
        y4 = pc_lgdp['u90']
        y5 = pc_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5)
    if i == 1:
        y1 = pc_lemp['b']
        y2 = pc_lemp['u95']
        y3 = pc_lemp['d95']
        y4 = pc_lemp['u90']
        y5 = pc_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5)

fig.text(0.5, 0.91, '(a) Estimates Using the GI Sentiment Index', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.62, '(b) Estimates Using the LSD Sentiment Index', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.33, '(b) Estimates Using the First Principal Component of the Sentiment Indexes', ha='center', fontsize=24,fontweight='bold')

plt.subplots_adjust(hspace=0.5)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixG.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%------------------Impulse Responses to Aggregate Shocks (Local Projections, Longer Horizon)----------------------
#-----------------------------------------------------------------------------------------------------------------------
lm_lgdp=pd.read_stata('Analysis/Local Projections/robustness/output/lm_lgdp_h36.dta')
lm_lemp=pd.read_stata('Analysis/Local Projections/robustness/output/lm_lemp_h36.dta')

rpu_lgdp=pd.read_stata('Analysis/Local Projections/robustness/output/rpu_lgdp_h36.dta')
rpu_lemp=pd.read_stata('Analysis/Local Projections/robustness/output/rpu_lemp_h36.dta')

# Define a function for subplots
def ax_plot(y1, y2, y3, y4, y5, ymin=-1, ymax=0.4):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8')
    ax.plot(x, y3, color='#C8C8C8')
    ax.plot(x, y4, color='#E8E8E8')
    ax.plot(x, y5, color='#E8E8E8')

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
    ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
    ax.axhline(y=0, color='black',linestyle='dotted',linewidth=2.5)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    ax.set_yticks(np.arange(ymin, ymax, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabels[i], fontsize=18)
    ax.set_xlabel('Months', fontsize=16)

    # title
    ax.set_title(titles[i], fontsize=20)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

#%% Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Plot
x=lm_lgdp['Years']

fig, axes = plt.subplots(2, 2, figsize=(18,12), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    if i==0:
        y1 = lm_lgdp['b']
        y2 = lm_lgdp['u95']
        y3 = lm_lgdp['d95']
        y4 = lm_lgdp['u90']
        y5 = lm_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5,ymin=-1.6)
    if i==1:
        y1 = lm_lemp['b']
        y2 = lm_lemp['u95']
        y3 = lm_lemp['d95']
        y4 = lm_lemp['u90']
        y5 = lm_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5)

for i, ax in enumerate(axes[1].flatten()):
    if i == 0:
        y1 = rpu_lgdp['b']
        y2 = rpu_lgdp['u95']
        y3 = rpu_lgdp['d95']
        y4 = rpu_lgdp['u90']
        y5 = rpu_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5,ymin=-1.6)
    if i == 1:
        y1 = rpu_lemp['b']
        y2 = rpu_lemp['u95']
        y3 = rpu_lemp['d95']
        y4 = rpu_lemp['u90']
        y5 = rpu_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig('Figures/Manuscript Figures - June 2025/irf_long_horizon.jpg', bbox_inches='tight')
plt.close()


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


#%% Appendix N: List of agencies, areas, and rule examples
agencies=pd.read_excel('Data/allUniqueAgencies.xlsx')
print(agencies.info())

agencies['area_name']=''
for i in range(0, len(agencies)):
    if np.isnan(agencies['area'][i])==False:
        area_code=str(int(agencies['area'][i]))
        agencies['area_name'][i]=dict_area[area_code]

agency_examples=agencies[agencies['area_name']!=''][['agency_name','department_name','area_name']].\
    sort_values(['area_name','department_name','agency_name']).reset_index(drop=True)

all_rules=pd.read_csv('Data/Regulatory Noun Chunks/ruleTitleNounChunksPhrases.csv')
print(all_rules.info())

rule_examples=all_rules[['rule_title','agency_name','department_name']].drop_duplicates(['agency_name','department_name'])
agency_examples=agency_examples.merge(rule_examples,on=['agency_name','department_name'],how='left')

agency_examples.to_csv('Figures/Manuscript Figures - June 2025/AppendixN.csv',index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------------------------------Interaction------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Appendix K: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Import IRF output
steps=12
lm_irf = pd.DataFrame()
lm_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/lm_'+v+'_interaction.dta')
    new = new.rename(columns={'Years': 'step'})
    for c in [c for c in new.columns if c!='step']:
        new=new.rename(columns={c:c+'_'+v})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)

rpu_irf = pd.DataFrame()
rpu_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata('Analysis/Local Projections/robustness/output/rpu_'+v+'_interaction.dta')
    new = new.rename(columns={'Years': 'step'})
    for c in [c for c in new.columns if c!='step']:
        new=new.rename(columns={c:c+'_'+v})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)

# Define a function for subplots
def ax_plot(y1,y2,y3,y4,y5,y6,legend1="Baseline",legend2="Robustness check"):
        ax.plot(x,y2,color='#dcdcdc')
        ax.plot(x,y3,color='#dcdcdc')
        ax.fill_between(x, y2, y3, facecolor='#dcdcdc')
        ax.plot(x,y1,color='black',linewidth=2.5,marker="D",label=legend1)
        ax.plot(x,y5,color=colors[1],linewidth=2.5)
        ax.plot(x,y6,color=colors[1],linewidth=2.5)
        ax.plot(x,y4,color=colors[1],linewidth=2.5,marker="o",label=legend2)

        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2.5)

        # title
        ax.set_title(titles[i], fontsize=20)

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,3))
        ax.set_yticks(np.arange(-1,0.5,0.2))
        ax.tick_params(axis='both',which='major',labelsize=14)
        ax.margins(x=0.01)

        # axis labels
        ax.set_ylabel(ylabels[i],fontsize=18)
        ax.set_xlabel('Months',fontsize=16)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

# Plot
x=lm_irf['step']

fig, axes = plt.subplots(2, 2, figsize=(18,14), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
        y1 = lm_irf['b_high_'+vars[i]]
        y2 = lm_irf['u95_high_'+vars[i]]
        y3 = lm_irf['d95_high_'+vars[i]]
        y4 = lm_irf['b_low_'+vars[i]]
        y5 = lm_irf['u95_low_'+vars[i]]
        y6 = lm_irf['d95_low_'+vars[i]]

        ax_plot(y1,y2,y3,y4,y5,y6,'Responses under High Uncertainty','Responses under Low Uncertianty')

for i, ax in enumerate(axes[1].flatten()):
        y1 = rpu_irf['b_high_'+vars[i]]
        y2 = rpu_irf['u95_high_'+vars[i]]
        y3 = rpu_irf['d95_high_'+vars[i]]
        y4 = rpu_irf['b_low_'+vars[i]]
        y5 = rpu_irf['u95_low_'+vars[i]]
        y6 = rpu_irf['d95_low_'+vars[i]]

        ax_plot(y1,y2,y3,y4,y5,y6,'Responses under High Sentiment','Responses under Low Sentiment')

#legend
axes[0][0].legend(loc=(0.5, -0.28), ncol=2, fontsize=16)
axes[1][0].legend(loc=(0.5, -0.28), ncol=2, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixK.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%---------------------------------Impulse Responses to Categorical Shocks (VAR)---------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
areas=pd.DataFrame()
areas['area_id']=list(dict_area.keys())
areas['area_title']=list(dict_area.values())

# Determine which human checking approach to use
robust='TotalOccurrence'
# Determine area classification name
area='DominantDistinctArea'

# Set number of steps for IRF
steps=60

vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# IRF - LM shocks
irf_lm_area=pd.DataFrame()
irf_lm_area['step']=range(0,steps+1)
for i in range(1, area_range):
        new=pd.read_stata('Analysis/VAR/irf_output/irf_lm_dda'+str(i)+'_'+robust.lower()+'.dta')
        new=new[new['irfname']=='baseline'][['step','oirfindexlgdp','llgdp','hlgdp','llgdp95','hlgdp95',
                                             'oirfindexlemp','llemp','hlemp','llemp95','hlemp95']].reset_index(drop=True)
        new=new.rename(columns={'oirfindexlgdp':'oirfindexlgdp'+str(i),'llgdp':'llgdp'+str(i),'hlgdp':'hlgdp'+str(i),
                                'llgdp95': 'llgdp95' + str(i), 'hlgdp95': 'hlgdp95' + str(i),
                                'oirfindexlemp':'oirfindexlemp'+str(i),'llemp':'llemp'+str(i),'hlemp':'hlemp'+str(i),
                                'llemp95':'llemp95'+str(i),'hlemp95':'hlemp95'+str(i)})
        irf_lm_area=irf_lm_area.merge(new,on='step',how='outer')

irf_lm_area=irf_lm_area[irf_lm_area['step']<=steps]

# IRF - RPU shocks
irf_rpu_area=pd.DataFrame()
irf_rpu_area['step']=range(0,steps+1)
for i in range(1, area_range):
        new=pd.read_stata('Analysis/VAR/irf_output/irf_rpu_dda'+str(i)+'_'+robust+'.dta')
        new=new[new['irfname']=='baseline'][['step','oirfindexlgdp','llgdp','hlgdp','llgdp95','hlgdp95',
                                             'oirfindexlemp','llemp','hlemp','llemp95','hlemp95']].reset_index(drop=True)
        new=new.rename(columns={'oirfindexlgdp':'oirfindexlgdp'+str(i),'llgdp':'llgdp'+str(i),'hlgdp':'hlgdp'+str(i),
                                'llgdp95':'llgdp95'+str(i),'hlgdp95':'hlgdp95'+str(i),
                                'oirfindexlemp':'oirfindexlemp'+str(i),'llemp':'llemp'+str(i),'hlemp':'hlemp'+str(i),
                                'llemp95':'llemp95'+str(i),'hlemp95':'hlemp95'+str(i)})
        irf_rpu_area=irf_rpu_area.merge(new,on='step',how='outer')

irf_rpu_area=irf_rpu_area[irf_rpu_area['step']<=steps]

#-----------------------------------------------------------------------------------------------------------------------
#%% Appendix Q1-2: Industrial Production Responses to a Negative Sentiment Shock By Regulatory Area (VAR)
x=irf_lm_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_lm_area['oirfindexlgdp' + str(i + 1)]
                y2 = irf_lm_area['llgdp95'+str(i + 1)]
                y3 = irf_lm_area['hlgdp95'+str(i + 1)]
                y4 = irf_lm_area['llgdp'+str(i + 1)]
                y5 = irf_lm_area['hlgdp'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='-')

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,6))
        ax.set_yticks(np.arange(-0.8,0.6,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[0], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ1-2.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%% Appendix Q2-2: Employment Responses to a Negative Sentiment Shock By Regulatory Area (VAR)
x=irf_lm_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_lm_area['oirfindexlemp' + str(i + 1)]
                y2 = irf_lm_area['llemp95'+str(i + 1)]
                y3 = irf_lm_area['hlemp95'+str(i + 1)]
                y4 = irf_lm_area['llemp'+str(i + 1)]
                y5 = irf_lm_area['hlemp'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='-')

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,6))
        ax.set_yticks(np.arange(-0.4,0.5,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[1], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ2-2.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%% Appendix Q3-2: Industrial Production Responses to an Uncertainty Shock By Regulatory Area (VAR)
x=irf_rpu_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_rpu_area['oirfindexlgdp' + str(i + 1)]
                y2 = irf_rpu_area['llgdp95'+str(i + 1)]
                y3 = irf_rpu_area['hlgdp95'+str(i + 1)]
                y4 = irf_rpu_area['llgdp'+str(i + 1)]
                y5 = irf_rpu_area['hlgdp'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='-')

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,6))
        ax.set_yticks(np.arange(-0.4,0.6,0.2))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[0], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ3-2.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%% Appendix Q4-2: Employment Responses to an Uncertainty Shock By Regulatory Area (VAR)
x=irf_rpu_area['step']

fig, axes = plt.subplots(5, 3, figsize=(40,35), sharex=True, sharey=False)

for i, ax in enumerate(axes.flatten()):
        if i+1 < area_range:
                y1 = irf_rpu_area['oirfindexlemp' + str(i + 1)]
                y2 = irf_rpu_area['llemp95'+str(i + 1)]
                y3 = irf_rpu_area['hlemp95'+str(i + 1)]
                y4 = irf_rpu_area['llemp'+str(i + 1)]
                y5 = irf_rpu_area['hlemp'+str(i + 1)]

                ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
                ax.plot(x, y2, color='#C8C8C8')
                ax.plot(x, y3, color='#C8C8C8')
                ax.plot(x, y4, color='#E8E8E8')
                ax.plot(x, y5, color='#E8E8E8')

                ax.set_title(dict_area[str(i + 1)], fontsize=44)
        else:
                break

        ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
        ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
        ax.axhline(y=0, color='black', linestyle='-')

        # ticks
        ax.set_xticks(np.arange(min(x),max(x)+1,6))
        ax.set_yticks(np.arange(-0.2,0.3,0.1))
        ax.tick_params(axis='both',which='major',labelsize=28)
        ax.margins(x=0.01)

        # borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

axes[3][2].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
fig.delaxes(axes[4,2])
fig.text(0.08, 0.5, ylabels[1], va='center', rotation='vertical', fontsize=44)
fig.text(0.5, 0.08, 'Months', ha='center', fontsize=44)
plt.subplots_adjust(hspace=0.2)

plt.savefig('Figures/Manuscript Figures - June 2025/AppendixQ4-2.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%---------------------------Forecast Error Variance Decomposition (FEVD) (VAR and LP)---------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Results generated using Gorodnichenko and Lee (2020) replication code

# Define results path
fevd_path='Data and Analysis/Analysis of Reg News/Analysis/Gorodnichenko and Lee 2020 Code/output'

# Define horizon
h=60

# Define confidence level
c=90

#-----------------------------------------------------------------------------------------------------------------------
#%% Output explained by regulatory uncertainty: VAR-based FEVD
df_vd_var=pd.DataFrame()
for ip in ['lm','rpu']:
    df_vd=pd.read_csv(f'{fevd_path}/lgdp/{ip.upper()}vdVAR_h{h}.csv',
                          header=None,index_col=False,names=[i for i in range(0,h+1)])
    df_vd=df_vd.T.rename(columns={0:f'{ip}_{ip}',1:f'{ip}_lsp',2:f'{ip}_ffr',3:f'{ip}_lemp',4:f'{ip}_lgdp'})

    df_vd_cb=pd.read_csv(f'{fevd_path}/lgdp/{ip.upper()}vdCbVAR_h{h}.csv',
                          header=None,index_col=False,names=[i for i in range(0,h*2+2)])
    df_vd_cb1=df_vd_cb[[i for i in range(0,h+1)]].T.\
        rename(columns={0:f'{ip}_{ip}_d{c}',1:f'{ip}_lsp_d{c}',2:f'{ip}_ffr_d{c}',
                        3:f'{ip}_lemp_d{c}',4:f'{ip}_lgdp_d{c}'}).\
        reset_index(drop=True)
    df_vd_cb2=df_vd_cb[[i for i in range(h+1,h*2+2)]].T.\
        rename(columns={0:f'{ip}_{ip}_u{c}',1:f'{ip}_lsp_u{c}',2:f'{ip}_ffr_u{c}',
                        3:f'{ip}_lemp_u{c}',4:f'{ip}_lgdp_u{c}'}).\
        reset_index(drop=True)

    df_vd_var=df_vd_var.merge(df_vd,left_index=True,right_index=True,how='outer').\
                merge(df_vd_cb1,left_index=True,right_index=True,how='outer').\
                merge(df_vd_cb2,left_index=True,right_index=True,how='outer')
print(df_vd_var.info())

#%% Define a function for subplots
def ax_plot(y1, y2, y3, ymin=0, ymax=0.2, yint=0.02):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8')
    ax.plot(x, y3, color='#C8C8C8')

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 4))
    ax.set_yticks(np.arange(ymin, ymax, yint))
    # ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel('Share of Variance Explained', fontsize=18)
    ax.set_xlabel('Months', fontsize=16)

    # title
    ax.set_title(titles[i], fontsize=20)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

#%% Plot
vars=['lm_lgdp','lm_lemp','rpu_lgdp','rpu_lemp']
titles=['Industrial Production', 'Employment','Industrial Production', 'Employment']

# Plot
x=[i for i in range(0,h+1)]

fig, axes = plt.subplots(2, 2, figsize=(18,12), sharex=False, sharey=False)

for i, ax in enumerate(axes.flatten()):
    y1 = df_vd_var[vars[i]]
    y2 = df_vd_var[f'{vars[i]}_u90']
    y3 = df_vd_var[f'{vars[i]}_d90']
    ax_plot(y1,y2,y3,0,0.24, 0.04)

fig.text(0.5, 0.93, '(a) VAR-based FEVD from a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) VAR-based FEVD from a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig(f'Data and Analysis/Analysis of Reg News/Figures/Manuscript Figures - June 2025/fevd_var_h{h}.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%% Output explained by regulatory uncertainty: LP-based FEVD
df_vd_lp=pd.DataFrame()
for rp in ['lgdp','lemp']:
    for ip in ['lm','rpu']:
        df_vd = pd.read_csv(f'{fevd_path}/{rp}/{ip.upper()}yvdBcR2_h{h}.csv',
                            header=None, index_col=False, names=[f'{ip}_{rp}'])

        df_vd_cb = pd.read_csv(f'{fevd_path}/{rp}/{ip.upper()}yvdBcCbR2_h{h}.csv',
                               header=None, index_col=False, names=[i for i in range(0, h + 1)])
        df_vd_cb = df_vd_cb.T.rename(columns={0: f'{ip}_{rp}_d{c}', 1: f'{ip}_{rp}_u{c}'})

        df_vd_lp=df_vd_lp.merge(df_vd,left_index=True,right_index=True,how='outer').\
                    merge(df_vd_cb,left_index=True,right_index=True,how='outer')
print(df_vd_lp.info())

#%% Plot
vars=['lm_lgdp','lm_lemp','rpu_lgdp','rpu_lemp']
titles=['Industrial Production', 'Employment','Industrial Production', 'Employment']

# Plot
x=[i for i in range(0,h+1)]

fig, axes = plt.subplots(2, 2, figsize=(18,12), sharex=False, sharey=False)

for i, ax in enumerate(axes.flatten()):
    y1 = df_vd_lp[vars[i]]
    y2 = df_vd_lp[f'{vars[i]}_u90']
    y3 = df_vd_lp[f'{vars[i]}_d90']
    ax_plot(y1,y2,y3,-0.1,0.7,0.1)

fig.text(0.5, 0.93, '(a) LP-based FEVD from a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) LP-based FEVD from a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig(f'Data and Analysis/Analysis of Reg News/Figures/Manuscript Figures - June 2025/fevd_lp_h{h}.jpg', bbox_inches='tight')
plt.close()
