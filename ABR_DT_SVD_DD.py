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

doc_complete = reviews_dd.text
doc_complete

# tokenize each review
corpus = [doc.split() for doc in doc_complete]
corpus

import gensim
from gensim import corpora, models

dictionary = corpora.Dictionary(corpus)

DFM = [dictionary.doc2bow(doc) for doc in corpus]
DFM

tfidf=models.TfidfModel(DFM)
DFM_tfidf=tfidf[DFM]

n_SVD=8

SVD_modeldd = models.LsiModel(DFM_tfidf, id2word=dictionary, num_topics=n_SVD)

SVD = SVD_modeldd[DFM_tfidf]

svd_array=gensim.matutils.corpus2csc(SVD).T.toarray()

svd_dfdd=pd.DataFrame(svd_array)
svd_dfdd

svd_dfdd.columns = ['SVD_1','SVD_2','SVD_3','SVD_4','SVD_5','SVD_6','SVD_7','SVD_8']
svd_dfdd

review_num_cols = reviews_dd[['rating','comments','Text_Len','Positivity','target']]
review_num_cols

model_dfdd = pd.concat([review_num_cols,svd_dfdd],axis=1)

model_dfdd

# X = all independent variables
Xdd = model_dfdd.drop('target', axis=1)  # replace 'target_column' with your actual target

# y = target variable
ydd = model_dfdd['target']

from sklearn.model_selection import train_test_split

Xdd_train, Xdd_test, ydd_train, ydd_test = train_test_split(
    Xdd, ydd, test_size=0.3, random_state=0  # adjust test_size as needed
)


from sklearn.tree import DecisionTreeClassifier
treedd = DecisionTreeClassifier(random_state=0,max_depth=4,min_samples_split=30)
treedd.fit(Xdd_train, ydd_train)
tree_predictions = treedd.predict(Xdd_test)
tree_predictions
treedd.score(Xdd_train,ydd_train)

from sklearn import tree 
from matplotlib import pyplot as plt 
plt.figure(figsize=[25,20])
tree.plot_tree(treedd,
feature_names=list(Xdd_train.columns.values),
class_names=True,
filled=True)
plt.show()

