#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------Appendix Figures------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
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

# Set directory
directory=os.path.dirname(os.path.realpath(__file__))

# # Create an output directory if it does not exist
output_folder=f'{directory}/../../../figures/appendix_figures'
os.makedirs(output_folder, exist_ok=True)


#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------Impulse Responses to Aggregate Shocks (Local Projections)----------------------------------
#-----------------------------------------------------------------------------------------------------------------------
lm_lgdp=pd.read_stata(f'{directory}/../../../output/lm_lgdp.dta')
lm_lemp=pd.read_stata(f'{directory}/../../../output/lm_lemp.dta')
gi_lgdp=pd.read_stata(f'{directory}/../../../output/gi_lgdp.dta')
gi_lemp=pd.read_stata(f'{directory}/../../../output/gi_lemp.dta')
lsd_lgdp=pd.read_stata(f'{directory}/../../../output/lsd_lgdp.dta')
lsd_lemp=pd.read_stata(f'{directory}/../../../output/lsd_lemp.dta')
pc_lgdp=pd.read_stata(f'{directory}/../../../output/pc_lgdp.dta')
pc_lemp=pd.read_stata(f'{directory}/../../../output/pc_lemp.dta')
rpu_lgdp=pd.read_stata(f'{directory}/../../../output/rpu_lgdp.dta')
rpu_lemp=pd.read_stata(f'{directory}/../../../output/rpu_lemp.dta')

# Define a function for subplots
def ax_plot(y1, y2, y3, y4, y5, title, ylabel, ymin=-1, ymax=0.4, ystep=0.2):
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
    ax.set_yticks(np.arange(ymin, ymax, ystep))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.margins(x=0.01)

    # axis labels
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel('Months', fontsize=16)

    # title
    ax.set_title(title, fontsize=20)

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
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])
    if i==1:
        y1 = gi_lemp['b']
        y2 = gi_lemp['u95']
        y3 = gi_lemp['d95']
        y4 = gi_lemp['u90']
        y5 = gi_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])

for i, ax in enumerate(axes[1].flatten()):
    if i == 0:
        y1 = lsd_lgdp['b']
        y2 = lsd_lgdp['u95']
        y3 = lsd_lgdp['d95']
        y4 = lsd_lgdp['u90']
        y5 = lsd_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])
    if i == 1:
        y1 = lsd_lemp['b']
        y2 = lsd_lemp['u95']
        y3 = lsd_lemp['d95']
        y4 = lsd_lemp['u90']
        y5 = lsd_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])

for i, ax in enumerate(axes[2].flatten()):
    if i == 0:
        y1 = pc_lgdp['b']
        y2 = pc_lgdp['u95']
        y3 = pc_lgdp['d95']
        y4 = pc_lgdp['u90']
        y5 = pc_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])
    if i == 1:
        y1 = pc_lemp['b']
        y2 = pc_lemp['u95']
        y3 = pc_lemp['d95']
        y4 = pc_lemp['u90']
        y5 = pc_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i])

fig.text(0.5, 0.91, '(a) Estimates Using the GI Sentiment Index', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.62, '(b) Estimates Using the LSD Sentiment Index', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.33, '(b) Estimates Using the First Principal Component of the Sentiment Indexes', ha='center', fontsize=24,fontweight='bold')

plt.subplots_adjust(hspace=0.5)

plt.savefig(f'{output_folder}/AppendixG.jpg', bbox_inches='tight')
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

#%% Appendix H1: GDP Responses to Regulatory Sentiment and Uncertainty Shocks (Quarterly)
steps=12
irf_gdp=pd.DataFrame()
irf_gdp['step']=range(0,steps+1)
for index in vars:
    new=pd.read_stata(f'{directory}/../../../output/'+index+'_lgdp_quarterly.dta')
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

        ax_plot(y1, y2, y3, y4,y5,title=titles[i],ylabel=ylabel,ymin=-1.5,ymax=0.6,ystep=0.5)

plt.subplots_adjust(hspace=0.4,wspace=0.2)

plt.savefig(f'{output_folder}/AppendixH1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix H2: Investment Responses to Regulatory Sentiment and Uncertainty Shocks (Quarterly)
steps=12
irf_investment=pd.DataFrame()
irf_investment['step']=range(0,steps+1)
for index in vars:
    new=pd.read_stata(f'{directory}/../../../output/'+index+'_lgross_quarterly.dta')
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

        ax_plot(y1, y2, y3, y4,y5,title=titles[i],ylabel=ylabel,ymin=-4,ymax=5,ystep=2)

plt.subplots_adjust(hspace=0.4,wspace=0.2)

plt.savefig(f'{output_folder}/AppendixH2.jpg', bbox_inches='tight')
plt.close()


#-----------------------------------------------------------------------------------------------------------------------
#%%---------------------------------Impulse Responses to Aggregate Shocks (VAR)-----------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Set some common variables
steps=36        # Number of steps for IRF

vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# IRF - LM
lm_irf=pd.read_stata(f'{directory}/../../../output/var_lm.dta')

lm_irf=lm_irf[lm_irf['step']<=steps]
lm_irf_baseline=lm_irf[lm_irf['irfname']=='baseline'].reset_index(drop=True)

# IRF - RPU
rpu_irf=pd.read_stata(f'{directory}/../../../output/var_rpu.dta')

rpu_irf=rpu_irf[rpu_irf['step']<=steps]
rpu_irf_baseline=rpu_irf[rpu_irf['irfname']=='baseline'].reset_index(drop=True)

#%% Appendix I1: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
# Plot
x=lm_irf_baseline['step']

fig, axes = plt.subplots(2, 2, figsize=(18,12), sharex=False, sharey=False)

for i, ax in enumerate(axes[0].flatten()):
    y1 = lm_irf_baseline['oirflm'+vars[i]]
    y2 = lm_irf_baseline['l'+vars[i]+'95']
    y3 = lm_irf_baseline['h'+vars[i]+'95']
    y4 = lm_irf_baseline['l'+vars[i]]
    y5 = lm_irf_baseline['h'+vars[i]]

    ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-0.8,ymax=0.4,ystep=0.2)

for i, ax in enumerate(axes[1].flatten()):
    y1 = rpu_irf_baseline['oirfrpu' + vars[i]]
    y2 = rpu_irf_baseline['l' + vars[i] + '95']
    y3 = rpu_irf_baseline['h' + vars[i] + '95']
    y4 = rpu_irf_baseline['l' + vars[i]]
    y5 = rpu_irf_baseline['h' + vars[i]]

    ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-0.6,ymax=0.8,ystep=0.2)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig(f'{output_folder}/AppendixI1.jpg', bbox_inches='tight')
plt.close()

#%% Appendix I2: Impulse Responses to a Regulatory Sentiment/Uncertainty Shock (Alternative VAR Specifications)
tests=['baseline', 'reverse', 'timetrend', 'vix', 'nsp', 'bi_output', 'rbi_output', 'bi_emp', 'rbi_emp']
tests_labels=['baseline','reverse','timetrend','vix','no s&p','bivariate','bivariate reverse']

# Define a function to plot
def ax_plot2(y1,y2,y3,y4,y5,y6,y7):
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

    ax_plot2(y1, y2, y3, y4, y5, y6, y7)

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

    ax_plot2(y1, y2, y3, y4, y5, y6, y7)

# legend
axes[0][0].legend(loc=(0.5, -0.36), ncol=4, fontsize=16)
axes[1][0].legend(loc=(0.5, -0.36), ncol=4, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig(f'{output_folder}/AppendixI2.jpg', bbox_inches='tight')
plt.close()


#-----------------------------------------------------------------------------------------------------------------------
#%%------------------Impulse Responses to Aggregate Shocks (Local Projections, Longer Horizon)----------------------
#-----------------------------------------------------------------------------------------------------------------------
lm_lgdp=pd.read_stata(f'{directory}/../../../output/lm_lgdp_h36.dta')
lm_lemp=pd.read_stata(f'{directory}/../../../output/lm_lemp_h36.dta')

rpu_lgdp=pd.read_stata(f'{directory}/../../../output/rpu_lgdp_h36.dta')
rpu_lemp=pd.read_stata(f'{directory}/../../../output/rpu_lemp_h36.dta')

# Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
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
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-1.6, ymax=0.4, ystep=0.2)
    if i==1:
        y1 = lm_lemp['b']
        y2 = lm_lemp['u95']
        y3 = lm_lemp['d95']
        y4 = lm_lemp['u90']
        y5 = lm_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-1, ymax=0.4, ystep=0.2)

for i, ax in enumerate(axes[1].flatten()):
    if i == 0:
        y1 = rpu_lgdp['b']
        y2 = rpu_lgdp['u95']
        y3 = rpu_lgdp['d95']
        y4 = rpu_lgdp['u90']
        y5 = rpu_lgdp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-1.6, ymax=0.6, ystep=0.2)
    if i == 1:
        y1 = rpu_lemp['b']
        y2 = rpu_lemp['u95']
        y3 = rpu_lemp['d95']
        y4 = rpu_lemp['u90']
        y5 = rpu_lemp['d90']
        ax_plot(y1,y2,y3,y4,y5,title=titles[i],ylabel=ylabels[i],ymin=-1, ymax=0.6, ystep=0.2)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.47, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.5)

plt.savefig(f'{output_folder}/AppendixJ.jpg', bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#%%--------------------------------------Control for other sentiment/uncertainty----------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

#%% Appendix K1: Impulse Responses to a Regulatory Sentiment Shock (Controlling for Economic Sentiment and Policy Uncertainty)
# Import IRF output
controls=['mich','newssent','vix','epu']
irf_gdp = pd.DataFrame()
irf_gdp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata(f'{directory}/../../../output/lm_lgdp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lgdp_' + c, 'u90': 'llgdp_' + c, 'd90': 'hlgdp_' + c,
                 'u95': 'llgdp95_' + c, 'd95': 'hlgdp95_' + c})
    irf_gdp = irf_gdp.merge(new, on='step').reset_index(drop=True)

irf_emp = pd.DataFrame()
irf_emp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata(f'{directory}/../../../output/lm_lemp_'+c+'.dta')
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
        ax_plot(y1, y2, y3, y4, y5, title=titles[i], ylabel=ylabels[i], ymin=-1, ymax=0.4, ystep=0.2)
        ax.set_ylabel(ylabels[i], fontsize=16)

fig.text(0.5, 0.9, '(a) Controlling for Michigan Consumer Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.69, '(b) Controlling for General Economic Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.48, '(c) Controlling for VIX', ha='center', fontsize=24)
fig.text(0.5, 0.27, '(d) Controlling for Economic Policy Uncertainty', ha='center', fontsize=24)
plt.subplots_adjust(hspace=0.55)

plt.savefig(f'{output_folder}/AppendixK1.jpg', bbox_inches='tight')

#%% Appendix J2: Impulse Responses to a Regulatory Uncertainty Shock (Controlling for Economic Sentiment and Policy Uncertainty)
# Import IRF output
controls=['mich','newssent','vix','epu']
irf_gdp = pd.DataFrame()
irf_gdp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata(f'{directory}/../../../output/rpu_lgdp_'+c+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': 'lgdp_' + c, 'u90': 'llgdp_' + c, 'd90': 'hlgdp_' + c,
                 'u95': 'llgdp95_' + c, 'd95': 'hlgdp95_' + c})
    irf_gdp = irf_gdp.merge(new, on='step').reset_index(drop=True)

irf_emp = pd.DataFrame()
irf_emp['step'] = range(0, steps + 1)
for c in controls:
    new = pd.read_stata(f'{directory}/../../../output/rpu_lemp_'+c+'.dta')
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
        ax_plot(y1, y2, y3, y4, y5,title=titles[i], ylabel=ylabels[i], ymin=-0.8, ymax=0.4, ystep=0.2)
        ax.set_ylabel(ylabels[i], fontsize=16)

fig.text(0.5, 0.9, '(a) Controlling for Michigan Consumer Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.69, '(b) Controlling for General Economic Sentiment', ha='center', fontsize=24)
fig.text(0.5, 0.48, '(c) Controlling for VIX', ha='center', fontsize=24)
fig.text(0.5, 0.27, '(d) Controlling for Economic Policy Uncertainty', ha='center', fontsize=24)
plt.subplots_adjust(hspace=0.55)

plt.savefig(f'{output_folder}/AppendixK2.jpg', bbox_inches='tight')


#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------------------------------Interaction------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Appendix L: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Import IRF output
steps=12
lm_irf = pd.DataFrame()
lm_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata(f'{directory}/../../../output/lm_'+v+'_interaction.dta')
    new = new.rename(columns={'Years': 'step'})
    for c in [c for c in new.columns if c!='step']:
        new=new.rename(columns={c:c+'_'+v})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)

rpu_irf = pd.DataFrame()
rpu_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata(f'{directory}/../../../output/rpu_'+v+'_interaction.dta')
    new = new.rename(columns={'Years': 'step'})
    for c in [c for c in new.columns if c!='step']:
        new=new.rename(columns={c:c+'_'+v})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)

# Define a function for subplots
def ax_plot3(y1,y2,y3,y4,y5,y6,legend1="Baseline",legend2="Robustness check"):
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

        ax_plot3(y1,y2,y3,y4,y5,y6,'Responses under High Uncertainty','Responses under Low Uncertianty')

for i, ax in enumerate(axes[1].flatten()):
        y1 = rpu_irf['b_high_'+vars[i]]
        y2 = rpu_irf['u95_high_'+vars[i]]
        y3 = rpu_irf['d95_high_'+vars[i]]
        y4 = rpu_irf['b_low_'+vars[i]]
        y5 = rpu_irf['u95_low_'+vars[i]]
        y6 = rpu_irf['d95_low_'+vars[i]]

        ax_plot3(y1,y2,y3,y4,y5,y6,'Responses under High Sentiment','Responses under Low Sentiment')

#legend
axes[0][0].legend(loc=(0.5, -0.28), ncol=2, fontsize=16)
axes[1][0].legend(loc=(0.5, -0.28), ncol=2, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig(f'{output_folder}/AppendixL.jpg', bbox_inches='tight')
plt.close()


#-----------------------------------------------------------------------------------------------------------------------
#%%----------------------------------------Removing Deregulation Articles-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#%% Appendix M: Impulse Responses to a Regulatory Sentiment or Uncertainty Shock
vars=['lgdp','lemp']
ylabels=['Industrial Production Response, %','Employment Response, %']
titles=['Industrial Production', 'Employment']

# Import IRF output
steps=12
lm_irf = pd.DataFrame()
lm_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata(f'{directory}/../../../output/lm_'+v+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v, 'u90': 'l'+v, 'd90': 'h' + v,
                 'u95': 'l' + v+'95', 'd95': 'h' + v+'95'})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)
    new = pd.read_stata(f'{directory}/../../../output/lm_'+v+'_nodereg.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v+'_nodereg', 'u90': 'l'+v+'_nodereg', 'd90': 'h' + v+'_nodereg',
                 'u95': 'l' + v+'95_nodereg', 'd95': 'h' + v+'95_nodereg'})
    lm_irf = lm_irf.merge(new, on='step').reset_index(drop=True)

rpu_irf = pd.DataFrame()
rpu_irf['step'] = range(0, steps + 1)
for v in vars:
    new = pd.read_stata(f'{directory}/../../../output/rpu_'+v+'.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v, 'u90': 'l'+v, 'd90': 'h' + v,
                 'u95': 'l' + v+'95', 'd95': 'h' + v+'95'})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)
    new = pd.read_stata(f'{directory}/../../../output/rpu_'+v+'_nodereg.dta')
    new = new.rename(
        columns={'Years': 'step', 'b': v+'_nodereg', 'u90': 'l'+v+'_nodereg', 'd90': 'h' + v+'_nodereg',
                 'u95': 'l' + v+'95_nodereg', 'd95': 'h' + v+'95_nodereg'})
    rpu_irf = rpu_irf.merge(new, on='step').reset_index(drop=True)

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

        ax_plot3(y1,y2,y3,y4,y5,y6,'Baseline: IRF to a Regulatory Sentiment Shock','Robustness check: removing "deregulat*" articles')

for i, ax in enumerate(axes[1].flatten()):
        y1 = rpu_irf[vars[i]]
        y2 = rpu_irf['l'+vars[i]+'95']
        y3 = rpu_irf['h'+vars[i]+'95']
        y4 = rpu_irf[vars[i]+'_nodereg']
        y5 = rpu_irf['l'+vars[i]+'95_nodereg']
        y6 = rpu_irf['h'+vars[i]+'95_nodereg']

        ax_plot3(y1,y2,y3,y4,y5,y6,'Baseline: IRF to a Regulatory Uncertainty Shock','Robustness check: removing "deregulat*" articles')

#legend
axes[0][0].legend(loc=(0.25, -0.28), ncol=2, fontsize=16)
axes[1][0].legend(loc=(0.25, -0.28), ncol=2, fontsize=16)

fig.text(0.5, 0.93, '(a) Impulse Responses to a Regulatory Sentiment Shock', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 0.44, '(b) Impulse Responses to a Regulatory Uncertainty Shock', ha='center', fontsize=24,fontweight='bold')
plt.subplots_adjust(hspace=0.7)

plt.savefig(f'{output_folder}/AppendixM.jpg', bbox_inches='tight')
plt.close()
