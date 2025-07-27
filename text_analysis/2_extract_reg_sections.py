# %%
import pandas as pd
import os
import re
import xml.etree.cElementTree as et
from lxml import etree

import spacy
from spacy.lang.en import English

# %%
df=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/parsed_xml_clean.pkl')
print(df.info())

# %%
# Check date range
print("Date range:",min(df['StartDate']), max(df['StartDate']))

# %%
# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    text=text.replace('\n',' ').replace('\r',' ')
    return text

# %%
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

# %%
# Function to print one XML example
def print_xml(ID):
    tree = etree.parse(filePath+ID+'.xml')
    xml = etree.tostring(tree, encoding="unicode", pretty_print=True)
    print(xml)

# %%
# Function to remove html tags from a string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# %%
# Function to identify the sentence with "*regulat*" and a sentence before and after (expanded regulatory sentences)
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
# Extract expanded regulatory sentences
regsents_expand=[]
for text in df['Text']:
    new=extractSentenceBeforeAfter(text)
    regsents_expand.append(new)
print(len(regsents_expand))

# %%
print(regsents_expand[0])
print(regsents_expand[-1])

# %%
df['RegSentsExpand']=regsents_expand

# %%
# Length of regulatory sections
df['RegSentExpandLength']=df['RegSentsExpand'].str.len()
print(df.sort_values('RegSentExpandLength',ascending=False)[['ID','RegSentExpandLength']].head(10))

# %%
print('# of articles with no "*regulat*" in full text:',df[df['RegSentExpandLength']==0]['ID'].nunique())

# %%
# Sort df
df=df.sort_values(['Newspaper','StartDate','Title']).reset_index(drop=True)
print(df.info())

# %%
df.drop(['TextLemmatized','Text','GroupNo'],axis=1).to_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/allRegSentsExpand.pkl')

# %%



