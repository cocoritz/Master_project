# %%
# Random Forest algorithm to classify network traffic 

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# %%
filepath = "/Users/colineritz/Desktop/Master_project/data_system/data analysis/NetScrapper-main/Dataset/KaggleImbalanced.csv" #import dat processed 
df = pd.read_csv(filepath) 

# %%
# df
# partial_data = df[:100000]
# df = partial_data

# %%
feats = [x for x in df.columns if x != 'ProtocolName']
X = df[feats]
Y = df['ProtocolName']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42) #split test and train set

# %%
# X_train
# y_train.unique()
# test_file= pd.merge(X_test,y_test,how = 'left',left_index = True, right_index = True)
# X_test.to_csv('test_file.csv', index=False)
# test_file.to_csv('test_file.csv', index=False)
# test_file.columns[:-1]

# %%
dt = DecisionTreeClassifier()
dt.fit(X_train , y_train)

# %%
dt.tree_.node_count, dt.tree_.max_depth

# %%
dt.score(X_test, y_test)

# %%
rf = RandomForestClassifier()

# %%
%%time
rf.fit(X_train , y_train)

# %%
# rf.tree_.node_count, rf.tree_.max_depth

# %%
%%time
rf.score(X_test, y_test)

# %%
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# model = RandomForestClassifier()
# param_grid = {'max_depth':[30,40,50],
#               'n_estimators':[20,30,40,50],  
#               'max_features':['auto','log2'],
#               'criterion':['gini','entropy']}
# GR = GridSearchCV(estimator = model, param_grid = param_grid, scoring='accuracy', cv=6)

# %%
# GR.fit(X, Y)

# %%
# GR.best_score_

# %%
# GR.best_params_

# %%
rf = RandomForestClassifier(max_depth=60, n_estimators=100, max_features='auto', criterion='entropy')

# %%
%%time
rf.fit(X_train, y_train)

# %%
%%time
rf.score(X_test, y_test)

# %%
class_list = df['ProtocolName'].unique()

# %%
import time

# %%
model_output = {}
for label in class_list:
    model_output.setdefault(label, [])
    myDataFrame = df[df['ProtocolName']==label]
    samples = len(myDataFrame)
    myFeats = [x for x in myDataFrame.columns if x != 'ProtocolName']
    X_features = myDataFrame[myFeats]
#     X_features = scaler.fit_transform(X_features)
    myLabel = myDataFrame['ProtocolName']
    tic = time.time()
#     my_predict = np.argmax(model.predict(X_features), axis=-1)
    predicted_class = rf.predict(X_features)
    toc = time.time()
    confidence_score = np.max(rf.predict(X_features))
#     predicted_class = encoder.inverse_transform(my_predict)
    time_taken = toc-tic
    my_acc = accuracy_score(myLabel, predicted_class) 
    model_output[label].append(predicted_class)
    model_output[label].append(time_taken)
    model_output[label].append(samples)
    model_output[label].append(my_acc)
    model_output[label].append(confidence_score)
    

# %%
print(model_output['GOOGLE_MAPS'])

# %%
with  open("Evaluation3.txt", 'w+') as f:
    for label in model_output.keys():
        f.write(label +"\t" + str(round(model_output[label][1], 2)) + "\t" + str(model_output[label][2]) + "\t" + str(round(model_output[label][3]*100, 2)) + "\n")
f.close()

# %%


# %%
%%time
y_pred = rf.predict(X_test)

# %%
print(classification_report(y_test, y_pred))

# %%
cf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(34,12)) 
sns.heatmap(cf_matrix,annot=True, ax=ax, fmt='d', annot_kws={"size": 12})
#plt.savefig("RF_cf2.png")

# %%
print(confusion_matrix(y_test, y_pred))

# %%
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [feats[i] for i in indices]
plt.figure(figsize=(24,12))
plt.title("Importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), names, rotation=90)
# plt.savefig("RF_features.png")

# %%
%%time
# print(indices)
# print(names)
# for feature in zip(feats, importances):
#     print(feature)
# print("--- %s seconds ---" % (time.time() - start_time))

# %%
for feature in zip(names, importances[indices]):
    print(feature)

# %%
# Threshold 0.01
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(rf, threshold=0.01)
sfm.fit(X_train, y_train)
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feats[feature_list_index])

# %%
selected_feat= X_train.columns[(sfm.get_support())]
# print(len(selected_feat))
print(selected_feat)

# %%
# from sklearn import tree
# import pydotplus
# estimator = rf.estimators_[7]
# dot_data = open("dtree.dot", 'w')
# tree.export_graphviz(estimator, out_file=dot_data)
# dot_data.close()
# dot_data = open("dtree.dot", 'r')
# dot_data = dot_data.read()
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("Rf_tree.png")

# %%
X = df[selected_feat]
Y = df['ProtocolName']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
rf = RandomForestClassifier(max_depth=60, n_estimators=100, max_features='auto', criterion='entropy')
rf.fit(X_train , y_train)
rf.score(X_test, y_test)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
cf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(34,12)) 
sns.heatmap(cf_matrix,annot=True, ax=ax, fmt='d', annot_kws={"size": 12})
plt.savefig("RF_cf_with_important_features.png")

# %%
print(confusion_matrix(y_test, y_pred))

# %%
# pipe1 = Pipeline([('rf', RandomForestClassifier(max_depth=60, n_estimators=100, max_features='auto', criterion='entropy'))])


# %%
from sklearn.neighbors import KNeighborsClassifier
pipe2 = Pipeline([('knn', KNeighborsClassifier(n_neighbors = 3, metric='manhattan', weights='distance'))])
pipe2.fit(X_train , y_train)
pipe2.predict(X_test)

# %%
from joblib import dump
# dump the pipeline model
dump(pipe2, filename="KNN_traffic_classification.joblib")

# %%
pipe.fit(X_train , y_train)

# %%
pipe.predict(X_train)

# %%
# from joblib import dump
# dump the pipeline model
dump(pipe, filename="RF_traffic_classification.joblib")

# %%
X_train.head()

# %%
pipe.predict(X_test)

# %%
X_test

# %%
len(X_test)

# %%
X_test['prediction'] = pipe.predict(X_test)
X_test.head()

# %%
# newTestData = X_test[:10]
X_test.to_csv('TestData2.csv', index=False)

# %%
newTestData.head()

# %%
newTestData.to_csv('TestData.csv', index=False)

# %%



