import pandas as pd
import os
import numpy as np

# Plotting Packages
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

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
output_folder=f'{directory}/../../figures/appendix_figures'
os.makedirs(output_folder, exist_ok=True)


#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------Impulse Responses to Categorical Shocks (Local Projections)------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# areas=pd.DataFrame()
# areas['area_id']=list(dict_area.keys())
# areas['area_title']=list(dict_area.values())

# Determine area classification name
# area='DominantDistinctArea'

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

#%% Appendix S1: Industrial Production Responses to a Negative Sentiment Shock By Regulatory Area
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

plt.savefig(f'{output_folder}/AppendixS1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix S2: Employment Responses to a Negative Sentiment Shock By Regulatory Area
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

plt.savefig(f'{output_folder}/AppendixS2.jpg', bbox_inches='tight')
plt.close()

#%% Appendix S3: Industrial Production Responses to an Uncertainty Shock By Regulatory Area
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

plt.savefig(f'{output_folder}/AppendixS3.jpg', bbox_inches='tight')
plt.close()

#%% Appendix S4: Employment Responses to an Uncertainty Shock By Regulatory Area
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

plt.savefig(f'{output_folder}/AppendixS4.jpg', bbox_inches='tight')
plt.close()
