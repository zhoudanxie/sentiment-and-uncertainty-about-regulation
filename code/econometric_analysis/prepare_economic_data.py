import pandas as pd
import datetime
import numpy as np
import os
from sklearn.decomposition import PCA

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))

# Create a data-for-analysis directory if it does not exist
output_folder=f'{directory}/../../data/processed_data/data_for_analysis'
os.makedirs(output_folder, exist_ok=True)

# Create an output folder for saving econometric results
os.makedirs(f'{directory}/../../output', exist_ok=True)

#-----------------------------------------------------------------------------------------------------------------------
#%% S&P 500
sp500=pd.read_csv(f'{directory}/../../data/raw_data/S&P500.csv')

sp500['Date']=sp500['Date'].astype('datetime64[ns]')
sp500['year']=sp500['Date'].dt.year
sp500['month']=sp500['Date'].dt.month

sp500=sp500.rename(columns={' Close':'sp500'})
sp500_monthly=sp500[['year','month','sp500']].groupby(['year','month']).agg('mean').reset_index()

sp500_monthly.to_stata(f'{output_folder}/sp_500_data.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% Macro data
# Industrial production
indus_prod=pd.read_csv(f'{directory}/../../data/raw_data/INDPRO.csv')

indus_prod['DATE']=indus_prod['DATE'].astype('datetime64[ns]')
indus_prod['year']=indus_prod['DATE'].dt.year
indus_prod['month']=indus_prod['DATE'].dt.month
indus_prod=indus_prod.rename(columns={'INDPRO':'indus_prod'})
indus_prod=indus_prod[['year','month','indus_prod']]

# Federal funds rate
fedfunds=pd.read_csv(f'{directory}/../../data/raw_data/FEDFUNDS.csv')

fedfunds['DATE']=fedfunds['DATE'].astype('datetime64[ns]')
fedfunds['year']=fedfunds['DATE'].dt.year
fedfunds['month']=fedfunds['DATE'].dt.month
fedfunds=fedfunds.rename(columns={'FEDFUNDS':'fedfundsrate'})
fedfunds=fedfunds[['year','month','fedfundsrate']]

# Employment
employment=pd.read_csv(f'{directory}/../../data/raw_data/PAYEMS.csv')

employment['DATE']=employment['DATE'].astype('datetime64[ns]')
employment['year']=employment['DATE'].dt.year
employment['month']=employment['DATE'].dt.month
employment=employment.rename(columns={'PAYEMS':'employment'})
employment=employment[['year','month','employment']]

# Real GDP
gdpc=pd.read_csv(f'{directory}/../../data/raw_data/GDPC1.csv')

gdpc['DATE']=gdpc['DATE'].astype('datetime64[ns]')
gdpc['year']=gdpc['DATE'].dt.year
gdpc['month']=gdpc['DATE'].dt.month
gdpc.loc[gdpc['month']==1,'quarter']=1
gdpc.loc[gdpc['month']==4,'quarter']=2
gdpc.loc[gdpc['month']==7,'quarter']=3
gdpc.loc[gdpc['month']==10,'quarter']=4
gdpc=gdpc.rename(columns={'GDPC1':'gdp'})
gdpc=gdpc[['year','quarter','gdp']]

# VIX
vix=pd.read_csv(f'{directory}/../../data/raw_data/VIX1990-2021.csv')
vix['Date']=vix['DATE'].astype('datetime64[ns]')
vix['year']=vix['Date'].dt.year
vix['month']=vix['Date'].dt.month
vix_monthly=vix[['year','month','CLOSE']].groupby(['year','month']).agg('mean').reset_index()
vix_monthly=vix_monthly.rename(columns={'CLOSE':'vix'})

vxo1=pd.read_excel(f'{directory}/../../data/raw_data/VXO1986-2003.xls',skiprows=2)
vxo1.loc[vxo1['Close']=='32.50  1.01','Close']=32.5     # Correct a data error
vxo1['Close']=vxo1['Close'].astype('float64')

vxo=vxo1[vxo1['Date']<=datetime.datetime(1989,12,31)].copy()
vxo['year']=vxo['Date'].dt.year
vxo['month']=vxo['Date'].dt.month
vxo_monthly=vxo[['year','month','Close']].groupby(['year','month']).agg('mean').reset_index()
vxo_monthly=vxo_monthly.rename(columns={'Close':'vix'})

vix_monthly=pd.concat([vxo_monthly,vix_monthly],axis=0).reset_index(drop=True)
vix_monthly=vix_monthly[['year','month','vix']]

# Merge all macro data
macro_data=indus_prod.merge(fedfunds,on=['year','month'],how='outer').merge(employment,on=['year', 'month'],how='outer').\
    merge(vix_monthly,on=['year','month'],how='outer').reset_index(drop=True)
macro_data.loc[macro_data['month']<=3, 'quarter']=1
macro_data.loc[(macro_data['month']>=4) & (macro_data['month']<=6), 'quarter']=2
macro_data.loc[(macro_data['month']>=7) & (macro_data['month']<=9), 'quarter']=3
macro_data.loc[(macro_data['month']>=10) & (macro_data['month']<=12), 'quarter']=4
macro_data=macro_data.merge(gdpc,on=['year','quarter'],how='outer').reset_index(drop=True)

macro_data.to_stata(f'{output_folder}/macro_data.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% NIPA
# Real Gross Private Domestic Investment
grossinvest=pd.read_csv(f'{directory}/../../data/raw_data/GPDIC1.csv')

grossinvest['DATE']=grossinvest['DATE'].astype('datetime64[ns]')
grossinvest['year']=grossinvest['DATE'].dt.year
grossinvest['month']=grossinvest['DATE'].dt.month
grossinvest.loc[grossinvest['month']==1,'quarter']=1
grossinvest.loc[grossinvest['month']==4,'quarter']=2
grossinvest.loc[grossinvest['month']==7,'quarter']=3
grossinvest.loc[grossinvest['month']==10,'quarter']=4
grossinvest=grossinvest.rename(columns={'GPDIC1':'grossinvestment'})
grossinvest=grossinvest[['year','quarter','grossinvestment']]

grossinvest.to_stata(f'{output_folder}/nipa.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% Michigan consumer sentiment
michsent=pd.read_csv(f'{directory}/../../data/raw_data/MICHSENT.csv',skiprows=1)
michsent=michsent[['Month','Year','Index']].rename(columns={'Month':'month','Year':'year','Index':'MonthlyMichiganIndexofConsume'})

michsent.to_stata(f'{output_folder}/consumer_sentiment_data.dta',write_index=False)

#%% BBD EPU
epu=pd.read_excel(f'{directory}/../../data/raw_data/EPU_BBD.xlsx')

epu=epu[epu['Month'].notnull()]
epu=epu.rename(columns={'Year':'year','Month':'month','News_Based_Policy_Uncert_Index':'epu'})
epu['year']=epu['year'].astype('int64')
epu['month']=epu['month'].astype('int64')
epu=epu[['year','month','epu']]

epu.to_stata(f'{output_folder}/epu.dta',write_index=False)

#%% Shapiro news sentiment
newssent=pd.read_excel(f'{directory}/../../data/raw_data/Shapiro_news_sentiment_data.xlsx', sheet_name='Data')

newssent['year']=newssent['date'].dt.year
newssent['month']=newssent['date'].dt.month
newssent_monthly=newssent[['year','month','News Sentiment']].groupby(['year','month']).agg('mean').reset_index()
newssent_monthly=newssent_monthly.rename(columns={'News Sentiment':'newssent'})

newssent_monthly.to_stata(f'{output_folder}/newssent.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% Aggregate regulatory indexes
# News attention index
regrelevance=pd.read_csv(f'{directory}/../../data/processed_data/news_attention_index.csv')

regrelevance[['year','month']]=regrelevance['year-month'].str.split('-',expand=True)
regrelevance['year']=regrelevance['year'].astype('int64')
regrelevance['month']=regrelevance['month'].astype('int64')
regrelevance=regrelevance[['year','month','RegRelevance']].copy()

# Sentiment indexes
regindex=pd.read_csv(f'{directory}/../../data/processed_data/aggregate_sentiment_indexes.csv')
regindex=regindex.rename(columns={'Year':'year','Month':'month'})

# Merge
regindex=regrelevance.merge(regindex,on=['year','month'],how='outer').reset_index(drop=True)

# Quarter
regindex.loc[regindex['month']<=3, 'quarter']=1
regindex.loc[(regindex['month']>=4) & (regindex['month']<=6), 'quarter']=2
regindex.loc[(regindex['month']>=7) & (regindex['month']<=9), 'quarter']=3
regindex.loc[(regindex['month']>=10) & (regindex['month']<=12), 'quarter']=4

# Add quarterly indexes
quarterly=pd.read_csv(f'{directory}/../../data/processed_data/aggregate_sentiment_indexes_quarterly.csv')

# Get PC of quarterly indexes
for dict in ['Uncertainty','GI','LM','LSD']:
    quarterly[dict+'Index_standardized']=(quarterly[dict+'Index']-np.mean(quarterly[dict+'Index']))/np.std(quarterly[dict+'Index'])

# PCA of standardized monthly sentiment indexes
features = ['GIIndex_standardized', 'LMIndex_standardized', 'LSDIndex_standardized']
x = quarterly.loc[:, features].values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['SentimentPC1_standardized', 'SentimentPC2_standardized'])
quarterly = pd.concat([quarterly, principalDf], axis = 1)

# Rename
quarterly=quarterly[['Year','quarter','LMIndex','GIIndex','LSDIndex','UncertaintyIndex','SentimentPC1_standardized']].\
        rename(columns={'Year':'year','LMIndex':'lm_qr','GIIndex':'gi_qr',
                        'LSDIndex':'lsd_qr','UncertaintyIndex':'rpu_qr',
                        'SentimentPC1_standardized':'pc_qr'})

# Merge
regindex=regindex.merge(quarterly,on=['year','quarter'],how='left')

# Export
regindex.to_stata(f'{output_folder}/regindex.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% Aggregate regulatory indexes with deregulation articles removed
regindex_dereg=pd.read_csv(f'{directory}/../../data/processed_data/aggregate_sentiment_indexes_nodereg.csv')
regindex_dereg=regindex_dereg.rename(columns={'Year':'year','Month':'month'})

# Export
regindex_dereg.to_stata(f'{output_folder}/regindex_nodereg.dta',write_index=False)

#-----------------------------------------------------------------------------------------------------------------------
#%% Categorical regulatory indexes
regindex_area=pd.read_csv(f'{directory}/../../data/processed_data/categorical_sentiment_indexes.csv')

area_range=15
for i in range(1,area_range):
    regindex_area=regindex_area.rename(columns={'Uncertainty_DominantDistinctArea'+str(i):'rpu_dda'+str(i),
                                                'GI_DominantDistinctArea'+str(i):'gi_dda'+str(i),
                                                'LM_DominantDistinctArea'+str(i):'lm_dda'+str(i),
                                                'LSD_DominantDistinctArea'+str(i):'lsd_dda'+str(i)})

regindex_area=regindex_area.rename(columns={'Year':'year','Month':'month'})
regindex_area.loc[regindex_area['month']<=3, 'quarter']=1
regindex_area.loc[(regindex_area['month']>=4) & (regindex_area['month']<=6), 'quarter']=2
regindex_area.loc[(regindex_area['month']>=7) & (regindex_area['month']<=9), 'quarter']=3
regindex_area.loc[(regindex_area['month']>=10) & (regindex_area['month']<=12), 'quarter']=4

regindex_area.to_stata(f'{output_folder}/regindex_area.dta',write_index=False)
