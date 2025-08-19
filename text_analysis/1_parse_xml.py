# %%
import xml.etree.cElementTree as et
import pandas as pd
import os
import re
import time
import datetime
import gc
import shutil

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Optional: Multiprocessing Module
# import multiprocessing as mp
# from multiprocessing import Pool
#
# # Check core count
# print(mp.cpu_count())

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
# directory="text_analysis"

# %% [markdown]
# ## 1. Parse XML

# %%
# Import updated data
filePath=f'{directory}/sample_data/'
files=[]
for file in os.listdir(filePath):
    if file.endswith(".xml"):
        files.append(file)
print('Total number of files in the sample data:', len(files))

# %%
# Function to remove html tags from a string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    return text

# %%
# Function to parse XML
def import_xml(filename):
    ID=filename.split('.xml')[0]
    file=filePath+filename
    
    xmlp = et.XMLParser(encoding="UTF-8")
    parsed_xml = et.parse(file,parser=xmlp)
    root = parsed_xml.getroot()
    
    try:
        for child in root.findall('Obj'):
            if child.find('Language').find('RawLang') != None:
                lang = child.find('Language').find('RawLang').text
            elif child.find('Language').find('ISO').find('ISOExpansion')!=None:
                lang = child.find('Language').find('ISO').find('ISOExpansion').text
            else:
                lang=''
        if lang=='English':
            for child in root.findall('Obj'):
                type=child.find('ObjectTypes').find('mstar').text
                title=child.find('TitleAtt').find('Title').text
                try:
                    startdate=child.find('StartDate').text
                    enddate=child.find('EndDate').text
                except:
                    startdate=child.find('NumericDate').text
                    enddate=child.find('NumericDate').text

            if root.find('TextInfo')!=None:
                for node in root.iter('Text'):
                    text=node.text
                    text=remove_spaces(remove_html_tags(text))
                    wordcount=node.get('WordCount')
            else:
                text=''
                wordcount=0

            for child in root.findall('DFS'):
                pubtitle=child.find('PubFrosting').find('Title').text
                sourcetype=child.find('PubFrosting').find('SourceType').text

            return (ID,title,type,startdate,enddate,text,wordcount,pubtitle,sourcetype)
        
        else:
            print(filename, ": non-English article")
    
    except:
        not_parsed.append(filename)
        print('Could not parse:',filename)

# %%
# Optional: use multiprocessing to parse all XMLs
# # Define a thread Pool to process multiple XML files simultaneously
# # Default set to 3, but may change number of processes depending on instance
# p = Pool(processes=8)
#
# # Apply function with Pool to corpus, may limit number of articles by using split
# start_time = time.time()
#
# not_parsed=[]
# processed_lists=p.map(import_xml, files)
#
# print("--- %s seconds ---" % (time.time() - start_time))

#%%
# Parse all XMLs
processed_lists=[]
not_parsed=[]
for file in files:
    processed_lists.append(import_xml(file))

if len(not_parsed)>0:
    print('Failed to parse XMLs:', len(not_parsed))
    print(not_parsed)

# %%
# Transform processed data into a dataframe
df = pd.DataFrame(processed_lists, columns=['ID','Title','Type','StartDate','EndDate','Text',
            'TextWordCount','PubTitle', 'SourceType'])

# %% [markdown]
# ## 2. Clean Data

# %%
# Check article type
# print(df['SourceType'].value_counts())
# print(df['Type'].value_counts())

# %%
# Include only Type==News
df=df[df['Type']=='News'].sort_values(['PubTitle','StartDate']).reset_index(drop=True)

# %%
# Convert dates
df['StartDate']=df['StartDate'].astype('datetime64[ns]')
df['Year']=df['StartDate'].astype('datetime64[ns]').dt.year
df['Month']=df['StartDate'].astype('datetime64[ns]').dt.month

# %%
# Check start and end dates for each pub title
# for title in df.sort_values('PubTitle')['PubTitle'].unique():
#     print(title,min(df[df['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date),
#          max(df[df['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date))

# %%
# Clean duplicated news articles due to overlapped databases (referring to ProQuest publication coverage)
df=df[~(((df['PubTitle']=='Boston Globe') & (df['StartDate']<datetime.datetime(1997,1,1)))
                  | ((df['PubTitle']=='Boston Globe (Online)') & (df['StartDate']<datetime.datetime(2015,9,8))))]
df=df[~(((df['PubTitle']=='Chicago Tribune') & (df['StartDate']<datetime.datetime(1996,12,4)))
                  | ((df['PubTitle']=='Chicago Tribune (Online)') & (df['StartDate']<datetime.datetime(2017,2,23))))]
df=df[~(((df['PubTitle']=='Los Angeles Times') & (df['StartDate']<datetime.datetime(1996,12,4)))
                  | ((df['PubTitle']=='Los Angeles Times (Online)') & (df['StartDate']<datetime.datetime(2017,2,21))))]
df=df[~((df['PubTitle']=='New York Times (Online)') & (df['StartDate']<datetime.datetime(1996,1,1)))]
df=df[~(((df['PubTitle']=='The Washington Post') & (df['StartDate']<datetime.datetime(1996,12,4)))
                  | ((df['PubTitle']=='The Washington Post (Online)') & (df['StartDate']<datetime.datetime(2016,5,21))))]
df=df[~(((df['PubTitle']=='USA TODAY') & (df['StartDate']<datetime.datetime(1997,2,17)))
                  | ((df['PubTitle']=='USA Today (Online)') & (df['StartDate']<datetime.datetime(2012,12,5))))]
df=df[~((df['PubTitle']=='Wall Street Journal (Online)') & (df['StartDate']<datetime.datetime(2010,1,8)))]

# %%
# Consolidate newspaper names
df.loc[(df['PubTitle']=='Boston Globe (pre-1997 Fulltext)') | (df['PubTitle']=='Boston Globe') | 
       (df['PubTitle']=='Boston Globe (Online)'),'Newspaper']='Boston Globe'
df.loc[(df['PubTitle']=='Wall Street Journal') | (df['PubTitle']=='Wall Street Journal (Online)'),
    'Newspaper']='Wall Street Journal'
df.loc[(df['PubTitle']=='USA TODAY (pre-1997 Fulltext)') | (df['PubTitle']=='USA TODAY') | 
       (df['PubTitle']=='USA Today (Online)'),'Newspaper']='USA Today'
df.loc[(df['PubTitle']=='Chicago Tribune (pre-1997 Fulltext)') | (df['PubTitle']=='Chicago Tribune') | 
       (df['PubTitle']=='Chicago Tribune (Online)'),'Newspaper']='Chicago Tribune'
df.loc[(df['PubTitle']=='Los Angeles Times') | (df['PubTitle']=='Los Angeles Times (pre-1997 Fulltext)') | 
        (df['PubTitle']=='Los Angeles Times (Online)'),'Newspaper']='Los Angeles Times'
df.loc[(df['PubTitle']=='New York Times') | (df['PubTitle']=='New York Times (Online)'),'Newspaper']='New York Times'
df.loc[(df['PubTitle']=='The Washington Post') | (df['PubTitle']=='The Washington Post (pre-1997 Fulltext)') | 
       (df['PubTitle']=='The Washington Post (Online)'),'Newspaper']='The Washington Post'

# %%
df=df.sort_values(['Newspaper','StartDate','Title']).reset_index(drop=True)

# %%
# Article count by pub title
# for title in df.sort_values('PubTitle')['PubTitle'].unique():
#     print(title,min(df[df['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date),
#          max(df[df['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date),':',
#          len(df[df['PubTitle']==title]))

# %%
# Article count by newspaper
# for title in df.sort_values('Newspaper')['Newspaper'].unique():
#     print(title,min(df[df['Newspaper']==title].sort_values('StartDate')['StartDate'].dt.date),
#          max(df[df['Newspaper']==title].sort_values('StartDate')['StartDate'].dt.date),':',
#          len(df[df['Newspaper']==title]))

# %% [markdown]
# ## 3. Identify and Remove Duplicated Articles

# %%
# Full text for certain articles is not available due to copyright restrictions
# print("Number of empty full texts:",df[df['Text']==""]['ID'].nunique())
# print(df[df['Text']==""]['Newspaper'].value_counts())
# # Examples
# print(df[df['Text']==""]['ID'][-10:])
# print(df[df['Text']==""]['Title'][-10:])

# %%
# Define a text preprocessor (lemmatizer)
def my_preprocessor(text):
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_punct | token.is_space]
    text_out=" ".join(lemmas)
    return text_out

# %%
# Convert ID and text to list
id_list=df['ID'].tolist()
text_list=df['Text'].tolist()
# print(len(text_list), len(id_list))

# %%
# Define a function to preprocess text by list index
def preprocess_text(i):
    id=id_list[i]
    text_out=my_preprocessor(text_list[i])
    return (id,text_out)

#%%
# Preprocess all texts
text_lemmatized=[]
for i in range(len(id_list)):
    text_lemmatized.append(preprocess_text(i))

# %%
# Optional: Use multipleprocessing to preprocess all texts
# start_time = time.time()
# with Pool(8) as p:
#     text_lemmatized=p.map(preprocess_text, list(range(len(id_list))))
# print("--- %s seconds ---" % (time.time() - start_time))

# %%
# Transform processed data into a dataframe
df_lemmatized = pd.DataFrame(text_lemmatized, columns=['ID','TextLemmatized'])
# print(df_lemmatized.info())

# %%
# Merge
df=df.merge(df_lemmatized, on='ID', how='left')

# %%
# Check duplicates
# df['GroupNo']=df.groupby('TextLemmatized').cumcount()+1
# print("Number of duplicated articles:",df[df['GroupNo']>1]['ID'].nunique())

# %%
# Keep the earliest article if duplicated
df_nodup=df.groupby('TextLemmatized').nth(0).reset_index()
# print(df_nodup.info())

# %%
# df_nodup['GroupNo']=df_nodup.groupby('TextLemmatized').cumcount()+1
# print("Number of duplicated articles:", df_nodup[df_nodup['GroupNo']>1]['ID'].nunique())
# print("Number of unavailable articles:",df_nodup[df_nodup['TextLemmatized']==""]['ID'].nunique())

# %%
# Remove dataframes to release memory
del df
del df_lemmatized
gc.collect()

# %%
# Check start and end dates for each newspaper
for title in df_nodup.sort_values('Newspaper')['Newspaper'].unique():
    print(title,min(df_nodup[df_nodup['Newspaper']==title].sort_values('StartDate')['StartDate'].dt.date),
         max(df_nodup[df_nodup['Newspaper']==title].sort_values('StartDate')['StartDate'].dt.date),':',
         len(df_nodup[df_nodup['Newspaper']==title]))

# %%
# Check start and end dates for each pub title
# for title in df_nodup.sort_values('PubTitle')['PubTitle'].unique():
#     print(title,min(df_nodup[df_nodup['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date),
#          max(df_nodup[df_nodup['PubTitle']==title].sort_values('StartDate')['StartDate'].dt.date),':',
#          len(df_nodup[df_nodup['PubTitle']==title]))

# %%
# Save cleaned data
output_folder=f'{directory}/sample_data/sample_output'
# If folder exists, delete it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Create output folder
os.makedirs(output_folder)

# Export output
df_nodup.to_pickle(f'{output_folder}/parsed_xml.pkl')



