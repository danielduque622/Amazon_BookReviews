import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

reviews_dd = pd.read_csv('data/Book_review.csv')

def dim():
  dimensions = len(set(reviews_dd['text'].str.split().explode().values))
  print(f'{dimensions} dimensions in the potential DFM.')
dim()

# top 20 words 
freq = pd.Series(' '.join(reviews_dd['text']).split()).value_counts()[:20]
freq

# convert all words into lower cases
reviews_dd['text'] = reviews_dd['text'].str.lower()

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')
reviews_dd['text'] = reviews_dd['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# top 20 words after removing stop words and making all words lower case
freq = pd.Series(' '.join(reviews_dd['text']).split()).value_counts()[:20]
freq
dim()

# remove punctuation
reviews_dd['text'] = reviews_dd['text'].str.replace(r'[^\w\s]+', '',regex=True)
dim()

# perform stremming
from nltk.stem import PorterStemmer
st = PorterStemmer()
reviews_dd['text'] = reviews_dd['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
dim()

# check top words
freq = pd.Series(' '.join(reviews_dd['text']).split()).value_counts()[:20]
freq

stop += ['make','get','would','well','also','way','it','first','want']
reviews_dd['text'] = reviews_dd['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# check top words
freq = pd.Series(' '.join(reviews_dd['text']).split()).value_counts()[:20]
freq

from wordcloud import WordCloud
comment_words = str(' '.join(reviews_dd['text']).split())

import string
comment_words = comment_words.translate(str.maketrans('','',string.punctuation))
wordcloud = WordCloud(background_color='white',
max_words=200,
width=1000,height=1000,
).generate(comment_words)
plt.figure(figsize=(8,8))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

doc_complete = reviews_dd.text
doc_complete

# tokenize each review
corpus = [doc.split() for doc in doc_complete]
corpus

import gensim
from gensim import corpora, models

dictionary = corpora.Dictionary(corpus)

from gensim.models import word2vec
modeldd = word2vec.Word2Vec(corpus, min_count=20,workers=3,window=3,sg=1)

modeldd.wv.most_similar('author',topn=6)

# stori is the root word for story
modeldd.wv.most_similar('stori',topn=6)

DFM = [dictionary.doc2bow(doc) for doc in corpus]
DFM

n_topics= n_topics=5
ldamodel = models.LdaModel(DFM,num_topics=n_topics,id2word=dictionary,passes=40)

import pyLDAvis
pyLDAvis.enable_notebook()
import pyLDAvis.gensim_models
vis = pyLDAvis.gensim_models.prepare(ldamodel,DFM,dictionary)
vis

