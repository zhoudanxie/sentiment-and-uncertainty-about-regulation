# %%
import pandas as pd
import os
import re
import xml.etree.cElementTree as et

import spacy
from spacy.lang.en import English

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
# directory="text_analysis"

# %%
df=pd.read_pickle(f'{directory}/sample_data/sample_output/parsed_xml.pkl')

# %%
# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    text=text.replace('\n',' ').replace('\r',' ')
    return text

# %%
# Set sentencizer
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe("sentencizer")

# %%
# Function to remove html tags from a string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# %%
# Function to identify the sentence with "*regulat*" and a sentence before and after (regulatory section)
def extractSentenceBeforeAfter(text):
    sentSet=set()
    text=remove_spaces(text)
    doc=nlp(text)
    sentList=list(doc.sents)
    for i in range(0, len(sentList)):
        sent=sentList[i].text.strip()
        if len(re.findall('regulat',sent,re.IGNORECASE))>0:
            sentSet.add(sent)
            if i>0:
                sentSet.add(sentList[i-1].text.strip())
            if i<len(sentList)-1:
                sentSet.add(sentList[i+1].text.strip())
    sentText=' '.join(sentSet)
    return sentText

# %%
# Extract regulatory sections
regsents_expand=[]
for text in df['Text']:
    new=extractSentenceBeforeAfter(text)
    regsents_expand.append(new)

df['RegSection']=regsents_expand

# %%
# Length of regulatory sections
df['RegSectionLength']=df['RegSection'].str.len()
print('# of articles with no "*regulat*" in full text:',df[df['RegSectionLength']==0]['ID'].nunique())

# %%
# Sort DF
df=df.sort_values(['Newspaper','StartDate','Title']).reset_index(drop=True)

# Export data
df.drop(['TextLemmatized','Text'],axis=1).to_pickle(f'{directory}/sample_data/sample_output/reg_sections.pkl')




