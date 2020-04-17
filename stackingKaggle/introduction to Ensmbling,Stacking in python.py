#URL : https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/data
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

#going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

#Load in the train and test tadasets
train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')

# Store out passenger ID for easy access
PassengerId = test['PassengerId']

train.head(3)
test.head(3)

full_data = [train, test]
full_data.head(3)
# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = train['Name'].apply(len)##check what is return value mean
train.head(3)
test.head(3)

train['Has_Cabin'] = train["Cabin"].apply(lambda x : 0 if type(x) == float else 1)
test['Has_Cabin'] = train["Cabin"].apply(lambda x : 0 if type(x) == float else 1)
train.head(3)
test.head(3)

# Feature engineering steps taken from sina
# Create new feature FamailySize as a combination of Sibsp and Parch
for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
#create a new feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

#Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
#Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset["Title"].replace('Mme', 'Mrs')

"""for dataset in full_data:
    #Mapping Sex
    dataset['Sex'] = dataset['Sex'].replace('female', 0)
    dataset['Sex'] = dataset['Sex'].replace('male', 1)
    #Mapping title
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Sex'] = dataset['Sex'].replace('Mr', 1)
    dataset['Sex'] = dataset['Sex'].replace('Miss', 2)
    dataset['Sex'] = dataset['Sex'].replace('Mrs', 3)
    dataset['Sex'] = dataset['Sex'].replace('Master', 4)
    dataset['Sex'] = dataset['Sex'].replace('Rare', 5)
    dataset['Title'] = dataset['Title'].fillna(0)
    #Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].replace({'S': 0})
    dataset['Embarked'] = dataset['Embarked'].replace({'C': 1})
    dataset['Embarked'] = dataset['Embarked'].replace({'Q': 2})
    #Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)"""

for dataset in full_data:
    #Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({"female" : 0, "male" : 1}).astype(int)

    #Mapping title
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    #Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2}).astype(int)

    #Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    #Feature model_selection 드랍오류잇음
    drop_elements = ['PassengerID', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis = 1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
    test = test.drop(drop_elements, axis = 1)

#Pearson Correlation Heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

##seaborn은 오류남
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
                        u'FamilySize',u'Title']], hue='Survived', palette = 'seismic', size=1.2, diag_kind = 'kde',
                        diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set forlds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x,y):
        return self.clf.fit(x,y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)

# Class to extend XGboost classifyer

#out of Fold Prediction
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

#Put in our parameters for said classifiers
#Random Forest parameters
rf_params = {
    'n_jobs' : -1,
    'n_estimators' : 500,
    'warm_start' : True,
    #'max_features': 0.2,
    'max_depth' : 6,
    'min_samples_leaf' : 2,
    'max_features' : 'sqrt',
    'verbose' : 0
}

#Extra Trees parameters
et_params = {
    'n_jobs' : -1,
    'n_estimators' : 500,
    #'max_features' : 0.5,
    'max_depth' : 8,
    'min_samples_leaf' : 2,
    'verbose' : 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators' : 500,
    'learning_rate' : 0.75
}

#Gradient Boosting parameters
gb_params = {
    'n_estimators' : 500,
    #'max_features' : 0.2,
    'max_depth' : 5,
    'min_samples_leaf' : 2,
    'verbose' : 0
}

#Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
}

# Create 5 objects that represent out 4 models
SEED = 0
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

#Creating Numpy array out of our train and test sets
#Creat Numpy array of train, test and target(Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Output of the First level predictions
# Create our OOF train and test predictions. these base results will be used as new Features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) #Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) #random forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) #AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test) #Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test) # Support Vector classifier

print("Training is complete")

#Feature importances generated from the different classifiers

rf_features = rf.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train, y_train)

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame({'features' : cols,
                                'Random Forest feature importances' : rf_features,
                                'Extra Trees feature importances' : et_features,
                                'AdaBoost feature importances' : ada_features,
                                'Gradient Boost feature importances' : gb_features
})

#interactive feature importances via Plotly scatterplots
#Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers' ,
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        #size = feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout=go.Layout(
    autosize=True,
    title = 'Random Forest Feature Importance',
    hovermode = 'closest',
#    xaxis = dict(
#    title = 'pop',
#    ticklen = 5,
#    zeroline = False,
#    gridwidth = 2,
#    ),
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Extra Trees Feature importances'].value,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size =25
        color =featured_dataframe['Extra Trees feature importances'].values,
        colorscale='Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'Extra Trees Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance'
        ticklen = 5,
        gridwidth =2
    ),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = 'scatter2010')

# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go. Layout(
    autosize = True,
    title = 'AdaBoost Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = 'scatter2010')

# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].value,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size =25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'Gradient Boosting Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, fimename = 'scatter2010')

# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis = 1) # axis = 1 computes the mean row-wise
feature_datafram.head(3)

#Plotly Barplot of Average Feature feature_importances

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x = x,
            y = y,
            width = 0.5
            marker = dict(
                color = feature_dataframe['mean'].values,
            colorscale = 'Portland',
            showscale = True,
            reversescale = False
            ),
            opacity = 0.6
)]

layout = go.Layout(
    autosize = True,
    title = 'Barplots of Mean Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)

# Second-level Predictions from the First-level Output
base_predictions_train = pd.DataFrame({'RandomForest' : rf_oof_train.ravel(),
    'ExtraTrees' : et_oof_train.ravel(),
    'Adaboost' : ada_oof_train.ravel(),
    'GradientBoost' : gb_oof_train.ravel()
    })
base_predictions_train.head()

#correlation Heatmap of the second Level Training set
data = [
    go.Heatmap(
        z = base_predictions_train.astype(float).corr().values,
        x = base_predictions_train.columns.values,
        y = base_predictions_train.columns.values,
            colorscale = 'Viridis',
            showscale = True,
            reversescale = True
    )
]
py.iplot(data, filename = 'labelled-heatmap')

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train, axis = 1))
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

#Second level learning model via XGBoost
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_extimators = 2000,
    max_depth = 4,
    min_child_weight = 2,
    #gamma =1,
    gamma = 0.9,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic',
    nthread = -1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


#Generate Submission file
StackingSubmission = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

"""Steps for Further Improvement
As a closing remark it must be noted that the steps taken above just show a very simple way of producing an ensemble stacker. You hear of ensembles created at the highest level of Kaggle competitions which involves monstrous combinations of stacked classifiers as well as levels of stacking which go to more than 2 levels.
Some additional steps that may be taken to improve one's score could be:
1. Implementing a good cross-validation strategy in training the models to find optimal parameter values
2. Introduce a greater variety of base models for learning. The more uncorrelated the results, the better the final score.""
