# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from statistics import mean 
from statistics import stdev

data = pd.read_csv('data.csv', sep = ',')
data.dropna(how="all", inplace = True)

data['Type'] = data['Type'].replace('Clean', 0)
data['Type'] = data['Type'].replace('Barrage', 1)
data['Type'] = data['Type'].replace('Tone', 2)
data['Type'] = data['Type'].replace('SP', 3) 
data['Type'] = data['Type'].replace('PA', 4)  

data_all = data
data_8   = data.drop(['Symbol time'], axis=1)
data_7   = data.drop(['Symbol time', 'Average Noise Power'], axis=1)

print('******* Model development with all features ******************')
X = data_all.iloc[:, 1:-1].values # features
y = data_all.iloc[:, 0].values    # label

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, 
                                                random_state = None)

#MLP=0.810129 with alpha=0.1, hll=150,150, max_iter=1500, solver=adam
#MLP=0.793028 with alpha=0.1,hll=70, max_iter=350, solver='adam'

models=[]
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
models.append(('LR', lrModel))

knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
models.append(('KNN', knnModel))

models.append(('NB', GaussianNB()))

dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
models.append(('DT', dtModel))

rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
models.append(('RFC', rfModel))

mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
models.append(('MLP', mlpModel)) 

results = {}
names   = []

for name, model in models:
    names.append(name)
    results[name] = []
    #print('************ %s ************'%name)
    for i in range(10):
        kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        cv_type = kfold
        scoring_metrics='accuracy'
        cv_results = cross_val_score(model, XTrain, yTrain, cv = cv_type, 
                                 scoring = scoring_metrics)
        #print('%.3f (%.3f)'%(cv_results.mean(), cv_results.std()))
        results[name].extend(list(cv_results))
    print('%s: %f (%f)' % (name, mean(results[name]), stdev(results[name])))

print("******* Predictions for LR ************")
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
lrModel.fit(XTrain, yTrain)
predictions = lrModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("******** Predictions for KNN *********")
knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
knnModel.fit(XTrain, yTrain)
predictions = knnModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("******** Predictions for NB ********")
nbModel = GaussianNB()
nbModel.fit(XTrain, yTrain)
predictions = nbModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("******* Predictions for DT ********")
dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
dtModel.fit(XTrain, yTrain)
predictions = dtModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********** Predictions for RF ***********")
rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
rfModel.fit(XTrain, yTrain)
predictions = rfModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 

print("******** Predictions for MLP ************")
mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
mlpModel.fit(XTrain, yTrain)
predictions = mlpModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 


print('******* Model development with 8 features ***************')
X = data_8.iloc[:, 1:-1].values # features
y = data_8.iloc[:, 0].values    # label

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, 
                                                random_state = None)

models=[]
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
models.append(('LR', lrModel))

knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
models.append(('KNN', knnModel))

models.append(('NB', GaussianNB()))

dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
models.append(('DT', dtModel))

rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
models.append(('RFC', rfModel))

mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
models.append(('MLP', mlpModel)) 

results = {}
names   = []

for name, model in models:
    names.append(name)
    results[name] = []
    #print('************ %s ************'%name)
    for i in range(10):
        kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        cv_type = kfold
        scoring_metrics='accuracy'
        cv_results = cross_val_score(model, XTrain, yTrain, cv = cv_type, 
                                 scoring = scoring_metrics)
        #print('%.3f (%.3f)'%(cv_results.mean(), cv_results.std()))
        results[name].extend(list(cv_results))
    print('%s: %f (%f)' % (name, mean(results[name]), stdev(results[name])))

print("********* Predictions for LR **********")
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
lrModel.fit(XTrain, yTrain)
predictions = lrModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********* Predictions for KNN ***********")
knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
knnModel.fit(XTrain, yTrain)
predictions = knnModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********* Predictions for NB *********")
nbModel = GaussianNB()
nbModel.fit(XTrain, yTrain)
predictions = nbModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("******** Predictions for DT *********")
dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
dtModel.fit(XTrain, yTrain)
predictions = dtModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("******** Predictions for RF *******")
rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
rfModel.fit(XTrain, yTrain)
predictions = rfModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 

print("******** Predictions for MLP *********")
mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
mlpModel.fit(XTrain, yTrain)
predictions = mlpModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 

print('******* Model development with 7 features ***********')
X = data_7.iloc[:, 1:-1].values # features
y = data_7.iloc[:, 0].values    # label

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, 
                                                random_state = None)

models=[]
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
models.append(('LR', lrModel))

knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
models.append(('KNN', knnModel))

models.append(('NB', GaussianNB()))

dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
models.append(('DT', dtModel))

rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
models.append(('RFC', rfModel))

mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
models.append(('MLP', mlpModel)) 

results = {}
names   = []

for name, model in models:
    names.append(name)
    results[name] = []
    #print('************ %s ************'%name)
    for i in range(10):
        kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        cv_type = kfold
        scoring_metrics='accuracy'
        cv_results = cross_val_score(model, XTrain, yTrain, cv = cv_type, 
                                 scoring = scoring_metrics)
        #print('%.3f (%.3f)'%(cv_results.mean(), cv_results.std()))
        results[name].extend(list(cv_results))
    print('%s: %f (%f)' % (name, mean(results[name]), stdev(results[name])))

print("********** Predictions for LR ***********")
lrModel = LogisticRegression(solver='liblinear', penalty='l1', C=100)
lrModel.fit(XTrain, yTrain)
predictions = lrModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********** Predictions for KNN **********")
knnModel = KNeighborsClassifier(weights='distance', n_neighbors=15, 
                                metric='manhattan')
knnModel.fit(XTrain, yTrain)
predictions = knnModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********* Predictions for NB *********")
nbModel = GaussianNB()
nbModel.fit(XTrain, yTrain)
predictions = nbModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********* Predictions for DT ***********")
dtModel = DecisionTreeClassifier(criterion='entropy', max_depth=20, 
                                 min_samples_leaf=3, min_samples_split=3)
dtModel.fit(XTrain, yTrain)
predictions = dtModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))

print("********* Predictions for RF *********")
rfModel = RandomForestClassifier(min_samples_leaf=1, min_samples_split=4, 
                                 n_estimators=500)
rfModel.fit(XTrain, yTrain)
predictions = rfModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 

print("******** Predictions for MLP *********")
mlpModel = MLPClassifier(alpha = 0.1, hidden_layer_sizes = (70,),
                         max_iter = 350, solver='adam')
mlpModel.fit(XTrain, yTrain)
predictions = mlpModel.predict(XTest)
print(accuracy_score(yTest, predictions))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions)) 
