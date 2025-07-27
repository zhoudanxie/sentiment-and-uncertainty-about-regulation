# %%
import pandas as pd
import re
# from nltk.corpus import stopwords
import pickle
import numpy as np
from string import punctuation
import time

# %%
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Import expanded reg sentences
df_regSentsExpand=pd.read_pickle('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/RegSentsExpand_NounChunks.pkl')
print(df_regSentsExpand.info())

# %%
# # Refine to reg relevant articles
# df=df_regSentsExpand[df_regSentsExpand['NounChunkMatchFiltered']>0].reset_index(drop=True)
df=df_regSentsExpand.reset_index(drop=True)
print(df.info())

# %%
# Negation words
negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
          "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]

# %%
# Function to negate
def negated(word):
    # Determine if preceding word is a negation word
    if word.lower() in negate:
        return True
    else:
        return False

# %%
# Function to tokenize
def tokenizer(text):
    doc=nlp(text)
    tokens=[token.text.lower() for token in doc if not token.is_punct | token.is_space]
    return tokens

# %%
# Function to lemmatize
def lemmatizer(text):
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_punct | token.is_space]
    return lemmas

# %% [markdown]
# ## 4.1. LM uncertainty

# %%
# LM dictionary
LMlist=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Aug2020/Sentiment Analysis/LoughranMcDonald_SentimentList.csv')
print(LMlist.info())

# %%
# LM uncertainty dictionary
LMuncertain=LMlist[LMlist['Uncertainty'].notnull()]['Uncertainty'].tolist()
uncertaindict={'Uncertainty': [w.lower() for w in LMuncertain]}
#print(uncertaindict, len(LMuncertain))

# %%
# Lemmatize LM uncertainty dictionary
uncertainset=set()
for w in uncertaindict['Uncertainty']:
    v=''.join(lemmatizer(w))
    uncertainset.add(v)
uncertainlist_lemmatized=list(uncertainset)
print(len(uncertainlist_lemmatized))

# %%
print(uncertainlist_lemmatized[0:20])

# %%
# Function to count uncertainty terms
def uncertainty_count(keywords_list, article):

    uncertain_count = 0
    uncertain_words = []
 
    input_words=lemmatizer(article)
    word_count = len(input_words)
    
    for i in range(0, word_count):
        if input_words[i] in keywords_list:
            uncertain_count += 1
            uncertain_words.append(input_words[i])
    
    results = [uncertain_count, uncertain_words]
 
    return results

# %%
# Run LM uncertainty through all expanded reg sentences
UncertaintyCount=[]
UncertaintyWords=[]
for text in df['RegSentsExpand']:
    results=uncertainty_count(uncertainlist_lemmatized, text)
    UncertaintyCount.append(results[0])
    UncertaintyWords.append(results[1])
print(len(UncertaintyCount))

# %%
df['UncertaintyCount']=UncertaintyCount
df['UncertaintyWords']=UncertaintyWords
print(df.info())

# %%
print(df[df['UncertaintyCount']!=0]['ID'].nunique())

# %%
print(df[['ID','UncertaintyCount','UncertaintyWords']].head(10))

# %%
df[['ID','UncertaintyCount','UncertaintyWords']].to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LMuncertainty.csv',index=False)

# %% [markdown]
# ## 4.2. LM sentiment

# %%
# Function to count sentiment terms
def sentiment_count(dict, article):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    """
    pos_count = 0
    neg_count = 0
 
    pos_words = []
    neg_words = []
 
    input_words=lemmatizer(article)
 
    word_count = len(input_words)
 
    for i in range(0, word_count):
        if input_words[i] in dict['Negative']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 0:
                neg_count += 1
                neg_words.append(input_words[i])
            
        if input_words[i] in dict['Positive']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
    '''
    print('The results with negation check:', end='\n\n')
    print('The # of positive words:', pos_count)
    print('The # of negative words:', neg_count)
    print('The list of found positive words:', pos_words)
    print('The list of found negative words:', neg_words)
    print('\n', end='')
    '''
    
    results = [word_count, pos_count, neg_count, pos_words, neg_words]
 
    return results

# %%
# LM sentiment dictionary
LMposWords=LMlist[LMlist['Positive'].notnull()]['Positive'].tolist()
LMnegWords=LMlist[LMlist['Negative'].notnull()]['Negative'].tolist()
print(len(LMnegWords),len(LMposWords))
print(LMnegWords[0:20],LMposWords[0:20])

# %%
# Lemmatize LM sentiment dictionary
LMnegset=set()
for w in LMnegWords:
    v=''.join(lemmatizer(w.lower()))
    LMnegset.add(v)
print(len(LMnegset))

LMposset=set()
for w in LMposWords:
    v=''.join(lemmatizer(w.lower()))
    LMposset.add(v)
print(len(LMposset))

LMdict={'Negative': list(LMnegset), 'Positive': list(LMposset)}

# %%
print(LMdict['Positive'][0:20])
print(LMdict['Negative'][0:20])

# %%
# Run LM sentiment through all expanded reg sentences
LMpositiveCount=[]
LMnegativeCount=[]
LMpositiveWords=[]
LMnegativeWords=[]
for text in df['RegSentsExpand']:
    results=sentiment_count(LMdict, text)
    LMpositiveCount.append(results[1])
    LMnegativeCount.append(results[2])
    LMpositiveWords.append(results[3])
    LMnegativeWords.append(results[4])

# %%
df['LMposCount']=LMpositiveCount
df['LMnegCount']=LMnegativeCount
df['LMposWords']=LMpositiveWords
df['LMnegWords']=LMnegativeWords

# %%
df[['ID','LMposCount','LMnegCount','LMposWords','LMnegWords']].to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LMsentiments.csv',index=False)

# %% [markdown]
# ## 4.3. GI sentiment

# %%
# Harvard GI sentiment dictionary
with open("/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Aug2020/Sentiment Analysis/GIposWords.txt", "rb") as fp:   # Unpickling
    GIposWords = pickle.load(fp)
with open("/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Aug2020/Sentiment Analysis/GInegWords.txt", "rb") as fp:   # Unpickling
    GInegWords = pickle.load(fp)

# %%
print(len(GIposWords),GIposWords[0:20])
print(len(GInegWords),GInegWords[0:20])

# %%
# Non-lemmetized version of GI dictionary
GIdict2={'Negative': [w.lower() for w in GInegWords], 'Positive': [w.lower() for w in GIposWords]}
print('Positive:',GIdict2['Positive'][0:20])
print('Negative:',GIdict2['Negative'][0:20])

# %%
# Run GI sentiment through all expanded reg sentences using non-lemmatized GI dictionary (performs better than lemmatized GI)
totalWordCount=[]
GIpositiveCount=[]
GInegativeCount=[]
GIpositiveWords=[]
GInegativeWords=[]
for text in df['RegSentsExpand']:
    results=sentiment_count(GIdict2, text)
    totalWordCount.append(results[0])
    GIpositiveCount.append(results[1])
    GInegativeCount.append(results[2])
    GIpositiveWords.append(results[3])
    GInegativeWords.append(results[4])

# %%
df['TotalWordCount']=totalWordCount
df['GIposCount']=GIpositiveCount
df['GInegCount']=GInegativeCount
df['GIposWords']=GIpositiveWords
df['GInegWords']=GInegativeWords

# %%
df[['ID','TotalWordCount','GIposCount','GInegCount','GIposWords','GInegWords']].to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/GIsentiments.csv',index=False)

# %% [markdown]
# ## 4.4. LSD sentiment

# %%
# Lexicoder Sentiment Dictionary (LSD)
LSDlist=pd.read_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Aug2020/Sentiment Analysis/LSDsentimentWords_wStar.csv')
print(LSDlist.info())

# %%
LSDneg=LSDlist[LSDlist['LSDnegative'].notnull()]['LSDnegative'].tolist()
LSDpos=LSDlist[LSDlist['LSDpositive'].notnull()]['LSDpositive'].tolist()
LSDdict={'Negative': [w.lower() for w in LSDneg], 'Positive': [w.lower() for w in LSDpos]}

# %%
# Seperate terms with & without stars in LSD dictionary
pos_star=[]
pos_nostar=[]
for m in LSDdict['Positive']:
    if "*" in m:
        m=m.replace('*','')
        pos_star.append(m)
    else:
        pos_nostar.append(m)
print(len(pos_star), len(pos_nostar))

neg_star=[]
neg_nostar=[]
for m in LSDdict['Negative']:
    if "*" in m:
        m=m.replace('*','')
        neg_star.append(m)
    else:
        neg_nostar.append(m)
print(len(neg_star), len(neg_nostar))

# %%
# Compile re patterns for terms with & without stars
pattern_pos_nostar=re.compile(r'\b(?:%s)\b' % '|'.join(pos_nostar))
pattern_pos_star=re.compile(r'\b(?:%s)[a-zA-Z]*\b' % '|'.join(pos_star))
pattern_neg_nostar=re.compile(r'\b(?:%s)\b' % '|'.join(neg_nostar))
pattern_neg_star=re.compile(r'\b(?:%s)[a-zA-Z]*\b' % '|'.join(neg_star))

# %%
# Function to count LSD sentiment terms
def LSDsentiment_count(dict, article):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    """
    pos_count = 0
    neg_count = 0
 
    pos_words = []
    neg_words = []
 
    input_words=tokenizer(article)    # No lemmatizing since LSD dictionary includes variations
 
    word_count = len(input_words)
 
    for i in range(0, word_count):
        find_neg=pattern_neg_nostar.findall(input_words[i])+pattern_neg_star.findall(input_words[i])
        if len(find_neg)>0:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 0:
                neg_count += 1
                neg_words.append(input_words[i])
        
        find_pos=pattern_pos_nostar.findall(input_words[i])+pattern_pos_star.findall(input_words[i])
        if len(find_pos)>0:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
    '''
    print('The results with negation check:', end='\n\n')
    print('The # of positive words:', pos_count)
    print('The # of negative words:', neg_count)
    print('The list of found positive words:', pos_words)
    print('The list of found negative words:', neg_words)
    print('\n', end='')
    '''
    
    results = [word_count, pos_count, neg_count, pos_words, neg_words]
 
    return results

# %%
# Run LSD sentiment through all expanded reg sentences
start_time = time.time()

LSDpositiveCount=[]
LSDnegativeCount=[]
LSDpositiveWords=[]
LSDnegativeWords=[]
failed=[]
for i in range(0, len(df['RegSentsExpand'])):
    try:
        results=LSDsentiment_count(LSDdict, df['RegSentsExpand'][i])
    except:
        results=[None, None, None, None, None]
        failed.append(i)        
        
    LSDpositiveCount.append(results[1])
    LSDnegativeCount.append(results[2])
    LSDpositiveWords.append(results[3])
    LSDnegativeWords.append(results[4])
print(len(failed))

print("--- %s seconds ---" % (time.time() - start_time))

# %%
print(len(failed))
print(len(LSDpositiveWords))

# %%
df['LSDposCount']=LSDpositiveCount
df['LSDnegCount']=LSDnegativeCount
df['LSDposWords']=LSDpositiveWords
df['LSDnegWords']=LSDnegativeWords

# %%
print(df.head())

# %%
for i in range(0,10):
    print(df['RegSentsExpand'][i],df['LSDposWords'][i],df['LSDnegWords'][i])

# %%
df[['ID','LSDposCount','LSDnegCount','LSDposWords','LSDnegWords']].to_csv('/home/ec2-user/SageMaker/New Uncertainty/Jan1985-Dec2021/LSDsentiments.csv',index=False)

# %%



