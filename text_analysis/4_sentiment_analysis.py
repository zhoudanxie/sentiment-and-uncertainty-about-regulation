# %%
import pandas as pd
import os
import re
import pickle
import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Set directory
directory=os.path.dirname(os.path.realpath(__file__))
# directory="text_analysis"

# %%
# Import regulatory sections
df=pd.read_pickle(f'{directory}/sample_data/sample_output/reg_sections.pkl')

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
LMlist=pd.read_csv(f'{directory}/supplementary_data/LoughranMcDonald_SentimentList.csv')

# %%
# LM uncertainty dictionary
LMuncertain=LMlist[LMlist['Uncertainty'].notnull()]['Uncertainty'].tolist()
uncertaindict={'Uncertainty': [w.lower() for w in LMuncertain]}

# %%
# Lemmatize LM uncertainty dictionary
uncertainset=set()
for w in uncertaindict['Uncertainty']:
    v=''.join(lemmatizer(w))
    uncertainset.add(v)
uncertainlist_lemmatized=list(uncertainset)

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
for text in df['RegSection']:
    results=uncertainty_count(uncertainlist_lemmatized, text)
    UncertaintyCount.append(results[0])
    UncertaintyWords.append(results[1])

df['UncertaintyCount']=UncertaintyCount
df['UncertaintyWords']=UncertaintyWords

# %%
print('Number of articles with uncertainty words:',df[df['UncertaintyCount']!=0]['ID'].nunique())

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

# %%
# Lemmatize LM sentiment dictionary
LMnegset=set()
for w in LMnegWords:
    v=''.join(lemmatizer(w.lower()))
    LMnegset.add(v)

LMposset=set()
for w in LMposWords:
    v=''.join(lemmatizer(w.lower()))
    LMposset.add(v)

LMdict={'Negative': list(LMnegset), 'Positive': list(LMposset)}

# %%
# Run LM sentiment through all expanded reg sentences
LMpositiveCount=[]
LMnegativeCount=[]
LMpositiveWords=[]
LMnegativeWords=[]
for text in df['RegSection']:
    results=sentiment_count(LMdict, text)
    LMpositiveCount.append(results[1])
    LMnegativeCount.append(results[2])
    LMpositiveWords.append(results[3])
    LMnegativeWords.append(results[4])

df['LMposCount']=LMpositiveCount
df['LMnegCount']=LMnegativeCount
df['LMposWords']=LMpositiveWords
df['LMnegWords']=LMnegativeWords

# %% [markdown]
# ## 4.3. GI sentiment

# %%
# Harvard GI sentiment dictionary
with open(f"{directory}/supplementary_data/GIposWords.txt", "rb") as fp:   # Unpickling
    GIposWords = pickle.load(fp)
with open(f"{directory}/supplementary_data/GInegWords.txt", "rb") as fp:   # Unpickling
    GInegWords = pickle.load(fp)

# %%
# Non-lemmetized version of GI dictionary
GIdict={'Negative': [w.lower() for w in GInegWords], 'Positive': [w.lower() for w in GIposWords]}

# %%
# Run GI sentiment through all expanded reg sentences using non-lemmatized GI dictionary (performs better than lemmatized GI)
totalWordCount=[]
GIpositiveCount=[]
GInegativeCount=[]
GIpositiveWords=[]
GInegativeWords=[]
for text in df['RegSection']:
    results=sentiment_count(GIdict, text)
    totalWordCount.append(results[0])
    GIpositiveCount.append(results[1])
    GInegativeCount.append(results[2])
    GIpositiveWords.append(results[3])
    GInegativeWords.append(results[4])

df['TotalWordCount']=totalWordCount
df['GIposCount']=GIpositiveCount
df['GInegCount']=GInegativeCount
df['GIposWords']=GIpositiveWords
df['GInegWords']=GInegativeWords

# %% [markdown]
# ## 4.4. LSD sentiment

# %%
# Lexicoder Sentiment Dictionary (LSD)
LSDlist=pd.read_csv(f'{directory}/supplementary_data/LSDsentimentWords_wStar.csv')

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

neg_star=[]
neg_nostar=[]
for m in LSDdict['Negative']:
    if "*" in m:
        m=m.replace('*','')
        neg_star.append(m)
    else:
        neg_nostar.append(m)

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
LSDpositiveCount=[]
LSDnegativeCount=[]
LSDpositiveWords=[]
LSDnegativeWords=[]

for text in df['RegSection']:
    results=LSDsentiment_count(LSDdict, text)
    LSDpositiveCount.append(results[1])
    LSDnegativeCount.append(results[2])
    LSDpositiveWords.append(results[3])
    LSDnegativeWords.append(results[4])

df['LSDposCount']=LSDpositiveCount
df['LSDnegCount']=LSDnegativeCount
df['LSDposWords']=LSDpositiveWords
df['LSDnegWords']=LSDnegativeWords

# %%
# Use filtered noun chunk matches to define reg relevance
df.loc[df['NounChunkMatchFiltered']>0,'RegRelevance']=1

# %%
# Calculate sentiment scores
df['UncertaintyScore']=df['UncertaintyCount']/df['TotalWordCount']*100
for dic in ['GI','LSD','LM']:
    df[dic+'score']=(df[dic+'posCount']-df[dic+'negCount'])/df['TotalWordCount']*100

# %%
# Export data
df.to_pickle(f'{directory}/sample_data/sample_output/sentiment_scores.pkl')




