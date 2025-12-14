#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------Main Paper Figures-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Plotting Packages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

import scipy.stats

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

# Set directory
directory=os.path.dirname(os.path.realpath(__file__))

# # Create an output directory if it does not exist
output_folder=f'{directory}/../../figures'
os.makedirs(output_folder, exist_ok=True)

#-----------------------------------------------------------------------------------------------------------------------
#%%------------------------------------------------News Attention to Regulation-------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Import data
df_att=pd.read_csv(f'{directory}/../../data/processed_data/news_attention_index.csv')

#%% Figure 1: Monthly Index of News Attention to Regulation
df_att['date']=df_att['year-month'].astype('datetime64[ns]').dt.date
x=df_att['date']
y=df_att['RegRelevance']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# events
ax.axvspan(datetime(1997,6,1),datetime(1997,7,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1997,6,1), 130, 'Tobacco Settlement', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), 62, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2002,7,1),datetime(2002,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2003,1,1), 137, 'Sarbanes-Oxley', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,10,1), 148, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 168, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,5,1), 175, 'Deepwater\nHorizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 190, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1), 175, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,11,1),datetime(2020,12,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,11,1), 186, '2020 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,12,1),datetime(2021,1,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,12,1), 173, 'COVID-19\nVaccines', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_ydata = lambda x: '$%1.2f' % x
ax.grid(False)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14, color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')

ax.set_ylabel('Index of News Attention to Regulation',fontsize=16)
ax.set_yticks(np.arange(50,max(y)+50,50))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig(f'{output_folder}/Figure1.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------Aggregate Regulatory Sentiment and Uncertainty Indexes-----------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Import indexes
monthlyIndex=pd.read_csv(f'{directory}/../../data/processed_data/aggregate_sentiment_indexes.csv')

monthlyIndex['Year-Month']=monthlyIndex['Year'].map(str)+'-'+monthlyIndex['Month'].map(str)
monthlyIndex['date']=monthlyIndex['Year-Month'].astype('datetime64[ns]').dt.date

#%% Correlations between sentiment indexes
print('LM & GI:',scipy.stats.pearsonr(monthlyIndex['LMIndex'], monthlyIndex['GIIndex']))
print('LM & LSD:',scipy.stats.pearsonr(monthlyIndex['LMIndex'], monthlyIndex['LSDIndex']))
print('LSD & GI',scipy.stats.pearsonr(monthlyIndex['LSDIndex'], monthlyIndex['GIIndex']))

print('LMstandardized & GIstandardized:',scipy.stats.pearsonr(monthlyIndex['LMIndex_standardized'], monthlyIndex['GIIndex_standardized']))
print('LMstandardized & LSDstandardized:',scipy.stats.pearsonr(monthlyIndex['LMIndex_standardized'], monthlyIndex['LSDIndex_standardized']))
print('LSDstandardized & GIstandardized',scipy.stats.pearsonr(monthlyIndex['LSDIndex_standardized'], monthlyIndex['GIIndex_standardized']))

#%% Figure 2: Monthly Index of Regulatory Sentiment
x=monthlyIndex['date']
y1=monthlyIndex['GIIndex_standardized']
y2=monthlyIndex['LSDIndex_standardized']
y3=monthlyIndex['LMIndex_standardized']
y4=monthlyIndex['SentimentPC1_standardized']

fig, ax = plt.subplots(1, figsize=(16,10))
ax.plot(x,y3,color=colors[2],linewidth=0.8,label='LM Sentiment Index')
ax.plot(x,y1,color=colors[1],linewidth=0.8,label='GI Sentiment Index')
ax.plot(x,y2,color='#FF4500',linewidth=0.8,label='LSD Sentiment Index')
ax.plot(x,y4,color='black',linewidth=1.2,label='First Principal Component')

# events
ax.axvspan(datetime(1990,9,1),datetime(1990,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1990,9,1), -4.8, 'Savings and\nLoan Crisis', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(1993,9,1),datetime(1993,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1993,9,1), 3.9, 'Clinton\nHealth Care Plan', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), -3.5, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2006,11,1),datetime(2006,12,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2006,11,1), 3.5, 'Bush\nMidterm Election', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), -2.6, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), -3.8, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,10,1), -5, 'Deepwater\nHorizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,12,1), -5.5, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2012,7,1),datetime(2012,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2012,7,1), -4, 'Libor\nScandal', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1), 4.4, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,3,1), -2.5, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,11,1),datetime(2020,12,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,11,1), 4.5, '2020 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2021,5,1),datetime(2021,6,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2021,5,1), 5.2, 'COVID-19\nVaccine', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_ydata = lambda x: '$%1.2f' % x

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Regulatory Sentiment Index',fontsize=16)
ax.set_yticks(np.arange(-6,7,2))
ax.set_ylim(bottom=-6)
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
# ax.spines['right'].set_color('#d3d3d3')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=14)
fig.subplots_adjust(bottom=0.15)

plt.savefig(f'{output_folder}/Figure2.jpg', bbox_inches='tight')
plt.close()

#%% Figure 3: Monthly Index of Regulatory Uncertainty
x=monthlyIndex['date']
y=monthlyIndex['UncertaintyIndex']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# events
ax.axvspan(datetime(1994,5,1),datetime(1994,6,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1994,5,1), 0.77, 'GAO Proposal\nfor Derivative\nRegulations', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), 0.75, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), 0.82, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 0.86, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,10,1), 0.875, 'Deepwater Horizon\nOil Spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 0.84, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),0.86, '2016 Presidential\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), 0.81, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_ydata = lambda x: '$%1.2f' % x

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Regulatory Uncertainty Index',fontsize=16)
ax.set_yticks(np.arange(round(min(y),1)-0.1,round(max(y),1)+0.1,0.1))
ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig(f'{output_folder}/Figure3.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------Categorical Regulatory Sentiment and Uncertainty Indexes---------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Import categorical indexes
sentiment_area=pd.read_csv(f'{directory}/../../data/processed_data/categorical_sentiment_indexes.csv')

sentiment_area['Year-Month']=sentiment_area['Year'].map(str)+'-'+sentiment_area['Month'].map(str)
sentiment_area['date']=sentiment_area['Year-Month'].astype('datetime64[ns]').dt.date

#%% Figure 5: Plot finance and banking regulation with rolling means
def plot_ax(x, y1,y2):
    ax.plot(x, y1, color=colors[0])
    ax.plot(x,y2,color=colors[7],linestyle='dashed',linewidth=2)

    # format the ticks
    years = mdates.YearLocator(2)  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years.
    datemin = np.datetime64(x.iloc[0], 'Y')
    datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    # Set tick and label format
    ax.tick_params(axis='both', which='major', labelsize=14, color='#d3d3d3')
    ax.tick_params(axis='both', which='minor', color='#d3d3d3')
    ax.grid(color='#d3d3d3', which='major', axis='y')

    # Borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d3d3d3')
    ax.spines['bottom'].set_color('#d3d3d3')

# Specify area
vars=['LM_DominantDistinctArea'+str(7),'Uncertainty_DominantDistinctArea'+str(7)]
ylabels=['Regulatory Sentiment Index','Regulatory Uncertainty Index']

# Calculate rolling means (12 months)
for i in range(len(vars)):
    sentiment_area[vars[i]+'_Mean']=sentiment_area[vars[i]].rolling(window=12,center=True).mean()

x=sentiment_area['date']

fig, axes = plt.subplots(2, 1, figsize=(14,15), sharex=False)

for i, ax in enumerate(axes.flatten()):
    y1=sentiment_area[vars[i]]
    y2=sentiment_area[vars[i]+'_Mean']
    plot_ax(x,y1,y2)
    ax.set_ylabel(ylabels[i], fontsize=16)
    if i==0:    #reg sentiment
        ax.set_yticks(np.arange(-3.5, 0, 0.5))

        # events
        ax.axvspan(datetime(1987, 11, 1), datetime(1987, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1987, 11, 1), -1, 'Vernon S&L\nCollapse', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(1990, 9, 1), datetime(1990, 10, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1990, 9, 1), -2.95, 'Savings and\nLoan Crisis', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(1998, 10, 1), datetime(1998, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1998, 10, 1), -2.7, 'LTCM\nCollapse', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2004, 9, 1), datetime(2004, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2004, 7, 1), -3.1, 'Fannie Mae\nAccounting-Rule\nViolation', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2008, 2, 1), datetime(2008, 3, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2008, 2, 1), -2.7, 'Bond\nInsurers', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2008, 9, 1), datetime(2008, 10, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2008, 9, 1), -3, 'Lehman\nBrothers', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2010, 1, 1), datetime(2010, 3, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2010, 2, 1), -1, 'Volcker\nRule', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2010, 7, 1), datetime(2010, 8, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2010, 7, 1), -1.3, 'Dodd-Frank', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2011, 10, 1), datetime(2011, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2011, 12, 1), -3.3, 'MF Global\nBankruptcy', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2012, 7, 1), datetime(2012, 8, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2012, 7, 1), -3, 'LIBOR\nScandal', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2016, 11, 1), datetime(2016, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2016, 11, 1), -0.85, '2016 Presidential\nElection', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2020, 4, 1), datetime(2020, 5, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2020, 4, 1), -2.6, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2020, 11, 1), datetime(2020, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2020, 11, 1), -0.75, '2020 Presidential\nElection', fontsize=13, color=colors[4], horizontalalignment='center')

    else:   #reg uncertainty
        ax.set_yticks(np.arange(0.2, 1.4, 0.2))

        # events
        ax.axvspan(datetime(1987, 11, 1), datetime(1987, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1987, 11, 1), 1, 'Vernon S&L\nCollapse', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(1990, 9, 1), datetime(1990, 10, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1990, 9, 1), 1.1, 'Savings and\nLoan Crisis', fontsize=13, color=colors[4],
                horizontalalignment='center')

        # ax.axvspan(datetime(1995, 1, 1), datetime(1995, 2, 1), alpha=0.5, color='#d3d3d3')
        # ax.text(datetime(1995, 1, 1), 1.05, '', fontsize=13, color=colors[4],
        #         horizontalalignment='center')

        ax.axvspan(datetime(1998, 10, 1), datetime(1998, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(1998, 10, 1), 0.95, 'LTCM\nCollapse', fontsize=13, color=colors[4],
                horizontalalignment='center')

        # ax.axvspan(datetime(2001, 11, 1), datetime(2001, 12, 1), alpha=0.5, color='#d3d3d3')
        # ax.text(datetime(2001, 11, 1), 1.05, '', fontsize=13, color=colors[4],
        #         horizontalalignment='center')

        ax.axvspan(datetime(2004, 9, 1), datetime(2004, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2004, 7, 1), 0.9, 'Fannie Mae\nAccounting-Rule\nViolation', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2008, 2, 1), datetime(2008, 3, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2008, 2, 1), 1.1, 'Bond\nInsurers', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2008, 9, 1), datetime(2008, 10, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2008, 9, 1), 1, 'Lehman\nBrothers', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2010, 1, 1), datetime(2010, 3, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2010, 2, 1), 1.13, 'Volcker\nRule', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2010, 7, 1), datetime(2010, 8, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2010, 7, 1), 0.55, 'Dodd-Frank', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2011, 10, 1), datetime(2011, 11, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2011, 12, 1), 1.05, 'MF Global\nBankruptcy', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2012, 7, 1), datetime(2012, 8, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2012, 7, 1), 0.45, 'LIBOR\nScandal', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2016, 11, 1), datetime(2016, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2016, 11, 1), 1, '2016 Presidential\nElection', fontsize=13, color=colors[4],
                horizontalalignment='center')

        ax.axvspan(datetime(2020, 4, 1), datetime(2020, 5, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2020, 4, 1), 0.95, 'Coronavirus\nOutbreak', fontsize=13, color=colors[4], horizontalalignment='center')

        ax.axvspan(datetime(2020, 11, 1), datetime(2020, 12, 1), alpha=0.5, color='#d3d3d3')
        ax.text(datetime(2020, 11, 1), 1.1, '2020 Presidential\nElection', fontsize=13, color=colors[4], horizontalalignment='center')

fig.text(0.5, 0.9, '(a) Regulatory Sentiment around Finance and Banking Regulation', ha='center', fontsize=20,fontweight='bold')
fig.text(0.5, 0.46, '(b) Regulatory Uncertainty around Finance and Banking Regulation', ha='center', fontsize=20,fontweight='bold')
plt.subplots_adjust(hspace=0.3)

plt.savefig(f'{output_folder}/Figure5.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------Impulse Responses to Aggregate Shocks (Local Projections)----------------------------------
#-----------------------------------------------------------------------------------------------------------------------
lm_lgdp=pd.read_stata(f'{directory}/../../output/lm_lgdp.dta')
lm_lemp=pd.read_stata(f'{directory}/../../output/lm_lemp.dta')
rpu_lgdp=pd.read_stata(f'{directory}/../../output/rpu_lgdp.dta')
rpu_lemp=pd.read_stata(f'{directory}/../../output/rpu_lemp.dta')

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

#%% Figure 4: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
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
        ax_plot(y1,y2,y3,y4,y5)
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
        ax_plot(y1,y2,y3,y4,y5)
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

plt.savefig(f'{output_folder}/Figure4.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------Impulse Responses to Categorical Shocks (Local Projections)------------------------------
#-----------------------------------------------------------------------------------------------------------------------
areas=pd.DataFrame()
areas['area_id']=list(dict_area.keys())
areas['area_title']=list(dict_area.values())

# Set number of steps for IRF
steps=12
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# IRF - LM shocks
irf_lm_area=pd.DataFrame()
irf_lm_area['step']=range(0,steps+1)
for i in range(1, area_range):
    new=pd.read_stata(f'{directory}/../../output/lm_dda'+str(i)+'_lgdp.dta')
    new=new.rename(columns={'Years':'step','b':'lgdp_'+str(i),'u90':'llgdp90_'+str(i),'d90':'hlgdp90_'+str(i),
                            'u95':'llgdp95_'+str(i),'d95':'hlgdp95_'+str(i)})
    irf_lm_area=irf_lm_area.merge(new,on='step',how='outer')

    new=pd.read_stata(f'{directory}/../../output/lm_dda'+str(i)+'_lemp.dta')
    new=new.rename(columns={'Years':'step','b':'lemp_'+str(i),'u90':'llemp90_'+str(i),'d90':'hlemp90_'+str(i),
                            'u95':'llemp95_'+str(i),'d95':'hlemp95_'+str(i)})
    irf_lm_area=irf_lm_area.merge(new,on='step',how='outer')

# IRF - RPU shocks
irf_rpu_area=pd.DataFrame()
irf_rpu_area['step']=range(0,steps+1)
for i in range(1, area_range):
    new=pd.read_stata(f'{directory}/../../output/rpu_dda'+str(i)+'_lgdp.dta')
    new=new.rename(columns={'Years':'step','b':'lgdp_'+str(i),'u90':'llgdp90_'+str(i),'d90':'hlgdp90_'+str(i),
                            'u95':'llgdp95_'+str(i),'d95':'hlgdp95_'+str(i)})
    irf_rpu_area=irf_rpu_area.merge(new,on='step',how='outer')

    new=pd.read_stata(f'{directory}/../../output/rpu_dda'+str(i)+'_lemp.dta')
    new=new.rename(columns={'Years':'step','b':'lemp_'+str(i),'u90':'llemp90_'+str(i),'d90':'hlemp90_'+str(i),
                            'u95':'llemp95_'+str(i),'d95':'hlemp95_'+str(i)})
    irf_rpu_area=irf_rpu_area.merge(new,on='step',how='outer')

#%% Figure 6: Impulse Responses for Selected Areas
# Define a function for subplots
def ax_plot(y1,y2,y3,y4,y5,ymin=-0.8,ymax=0.8):
    ax.plot(x, y1, color='black', linewidth=2.5, marker="D")
    ax.plot(x, y2, color='#C8C8C8',linewidth=0.1)
    ax.plot(x, y3, color='#C8C8C8',linewidth=0.1)
    ax.plot(x, y4, color='#E8E8E8',linewidth=0.1)
    ax.plot(x, y5, color='#E8E8E8',linewidth=0.1)

    ax.set_title(dict_area[str(area)], fontsize=34)

    ax.fill_between(x, y2, y3, facecolor='#C8C8C8')
    ax.fill_between(x, y4, y5, facecolor='#E8E8E8')
    ax.axhline(y=0, color='black',linestyle='dotted',linewidth=2.5)

    # ticks
    ax.set_xticks(np.arange(min(x), max(x) + 1, 3))
    ax.set_yticks(np.arange(ymin, ymax, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.margins(x=0.01)

    # borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

# Select areas:
lmlgdp_areas=[1,3,7]
lmlemp_areas=[1,3,7]
rpulgdp_areas=[6,8,7]
rpulemp_areas=[6,8,7]

# Plot
x=irf_lm_area['step']

fig, axes = plt.subplots(4, 3, figsize=(30,26), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    area=lmlgdp_areas[i]
    y1 = irf_lm_area['lgdp_' + str(area)]
    y2 = irf_lm_area['llgdp95_'+str(area)]
    y3 = irf_lm_area['hlgdp95_'+str(area)]
    y4 = irf_lm_area['llgdp90_'+str(area)]
    y5 = irf_lm_area['hlgdp90_'+str(area)]

    ax_plot(y1,y2,y3,y4,y5,-1,0.4)
    # axis labels
    ax.set_ylabel(ylabels[0], fontsize=24)

for i, ax in enumerate(axes[1].flatten()):
    if i<len(lmlemp_areas):
        area = lmlemp_areas[i]
        y1 = irf_lm_area['lemp_' + str(area)]
        y2 = irf_lm_area['llemp95_' + str(area)]
        y3 = irf_lm_area['hlemp95_' + str(area)]
        y4 = irf_lm_area['llemp90_' + str(area)]
        y5 = irf_lm_area['hlemp90_' + str(area)]

        ax_plot(y1, y2, y3, y4, y5, -0.4, 0.6)
        # axis labels
        ax.set_ylabel(ylabels[1], fontsize=24)

for i, ax in enumerate(axes[2].flatten()):
    if i<len(rpulgdp_areas):
        area=rpulgdp_areas[i]
        y1 = irf_rpu_area['lgdp_' + str(area)]
        y2 = irf_rpu_area['llgdp95_' + str(area)]
        y3 = irf_rpu_area['hlgdp95_' + str(area)]
        y4 = irf_rpu_area['llgdp90_' + str(area)]
        y5 = irf_rpu_area['hlgdp90_' + str(area)]

        ax_plot(y1,y2,y3,y4,y5,-0.8,0.6)
        # axis labels
        ax.set_ylabel(ylabels[0], fontsize=24)

for i, ax in enumerate(axes[3].flatten()):
    if i<len(rpulemp_areas):
        area=rpulemp_areas[i]
        y1 = irf_rpu_area['lemp_' + str(area)]
        y2 = irf_rpu_area['llemp95_' + str(area)]
        y3 = irf_rpu_area['hlemp95_' + str(area)]
        y4 = irf_rpu_area['llemp90_' + str(area)]
        y5 = irf_rpu_area['hlemp90_' + str(area)]

        ax_plot(y1,y2,y3,y4,y5,-0.4,0.6)
        # axis labels
        ax.set_ylabel(ylabels[0], fontsize=24)

fig.text(0.5, 0.08, 'Months', ha='center', fontsize=24)
fig.text(0.5, 0.91, '(a) Output Responses to a Regulatory Sentiment Shock', ha='center', fontsize=38,fontweight='bold')
fig.text(0.5, 0.7, '(b) Employment Responses to a Regulatory Sentiment Shock', ha='center', fontsize=38,fontweight='bold')
fig.text(0.5, 0.49, '(c) Output Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=38,fontweight='bold')
fig.text(0.5, 0.28, '(d) Employment Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=38,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig(f'{output_folder}/Figure6.jpg', bbox_inches='tight')
plt.close()

#%% End
print(f"All main figures are saved in the {output_folder} folder.")