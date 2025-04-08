import numpy as np
import pandas as pd

df_dd = pd.read_csv('data/Book_review.csv')

df_dd.describe(include="all")

df_dd.head()

df_dd.dtypes

X_dd = df_dd[['rating','comments','Text_Len','Positivity']]
y_dd = df_dd['target']

from sklearn.model_selection import train_test_split

Xdd_train, Xdd_val, ydd_train, ydd_val = train_test_split(X_dd,y_dd, test_size = 0.3, random_state=0)

ydd_train.head()

from sklearn import tree 
from sklearn.tree import export_text
dtreedd = tree.DecisionTreeClassifier(max_depth=4,min_samples_split=30)
dtreedd = dtreedd.fit(Xdd_train,ydd_train)

r = export_text(dtreedd, feature_names=list(Xdd_train.columns.values))
print(r)

from matplotlib import pyplot as plt 
plt.figure(figsize=[25,20])
tree.plot_tree(dtreedd,
feature_names=list(Xdd_train.columns.values),
class_names=True,
filled=True)
plt.show()

dtreedd.score(Xdd_val,ydd_val)

