import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import sys

def preprocess(data, is_train):
    global age_imputer, fare_imputer, fare_pt, num_scaler

    # Remove features that don't matter
    data = data.drop(['PassengerId', 'Name'], axis=1)
    # These might matter but we'd have to figure out how to extract numeric features from them
    data = data.drop(['Ticket', 'Cabin'], axis=1)

    # One-hot encode categorical variables
    data = pd.get_dummies(data)

    # Drop highly correlated features
    if is_train:
        data = data.drop('Survived_0', axis=1)
    data = data.drop('Sex_female', axis=1)

    if is_train:
        # Remove outliers
        data = data.drop(data[data['Fare'] > 500].index)

    if is_train:
        # XXX: in the future, consider IterativeImputer
        age_imputer = SimpleImputer(strategy='mean').fit(data['Age'].values.reshape(-1, 1))
        fare_imputer = SimpleImputer(strategy='median').fit(data['Fare'].values.reshape(-1, 1)) # not normally distributed, mean may not be representative
    data['Age'] = age_imputer.transform(data['Age'].values.reshape(-1, 1))
    data['Fare'] = fare_imputer.transform(data['Fare'].values.reshape(-1, 1))

    if is_train:
        fare_pt = PowerTransformer().fit(data['Fare'].values.reshape(-1, 1))
    data['Fare_Pt'] = fare_pt.transform(data['Fare'].values.reshape(-1, 1))
    data = data.drop('Fare', axis=1)
    
    # Scale numerical attributes before feeding them to machine learning algorithms
    num_attribs = ['Age', 'Fare_Pt', 'SibSp', 'Parch']
    if is_train:
        num_scaler = StandardScaler().fit(data[num_attribs])
    data[num_attribs] = num_scaler.transform(data[num_attribs])

    return data

def display_metrics(y_true, y_pred, mtype):
    print("accuracy:", accuracy_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred))
    print("recall:", recall_score(y_true, y_pred))
    print("f1:", f1_score(y_true, y_pred))

def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'all'
    cv = 'cv' in sys.argv[1:]

    raw_train = pd.read_csv('titanic/train.csv', dtype={'Survived': 'category', 'Pclass': 'category', 'Sex': 'category'})
    raw_test = pd.read_csv('titanic/test.csv', dtype={'Pclass': 'category', 'Sex': 'category'})

    train = preprocess(raw_train, is_train=True)
    test = preprocess(raw_test, is_train=False)

    X_train, y_train = train.drop('Survived_1', axis=1), train['Survived_1']
    X_test = test

    models = {
        'tree': DecisionTreeClassifier(random_state=42),
        'forest': RandomForestClassifier(random_state=42),
        'logistic': LogisticRegression(random_state=42),
        'linearsvc': LinearSVC(dual=False),
        'sgd': SGDClassifier(random_state=42),
        'svc': SVC(),
        'kneighbors': KNeighborsClassifier(),
        #'gaussiannb': GaussianNB(), # not promising
        #'perceptron': Perceptron() # not promising
    }

    param_grids = {
        'tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(np.arange(3, 8)) + [None]
        },
        'forest': {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_depth': list(np.arange(3, 8)) + [None]
        },
        'logistic': {
            'penalty': ['l2'], # only l2, none are supported by the default solver
            'C': 10. ** np.arange(-4, 5)
        },
        'linearsvc': {
            'penalty': ['l2', 'l1'],
            'loss': ['squared_hinge'], # hinge is only supported when dual=True?
            'C': 10. ** np.arange(-3, 0) # C=1 or above with l1 penalty causes ConvergenceWwarning
        },
        'sgd': [{
            'penalty': ['l2', 'l1'],
            'loss': ['hinge', 'log', 'modified_huber', 'perceptron'], # squared_hinge causes ConvergenceWarning
            'alpha': 10. ** np.arange(-10, 1)
        }, {
            'penalty': ['elasticnet'],
            'loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
            'alpha': 10. ** np.arange(-10, 1),
            'l1_ratio': np.linspace(.1, .9, num=9)
        }],
        'kneighbors': {
            'n_neighbors': np.arange(1, 11),
            'weights': ['uniform', 'distance'],
            'leaf_size': np.arange(3, 11),
            'p': [1, 2]
        }
    }

    for (mtype, model) in models.items():
        if model_type != 'all' and mtype != model_type:
            continue
        print("Using model", mtype)

        if mtype == 'svc' or not cv: # trying to grid search svc takes a *very* long time
            model.fit(X_train, y_train)
        else:
            print("(cv) Starting grid search")
            grid_search = GridSearchCV(model, param_grids[mtype], cv=10, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print("(cv) Finished grid search")
            print("(cv) Best params:", grid_search.best_params_)
            print("(cv) Best score:", grid_search.best_score_)

        y_pred_train = model.predict(X_train)
        print("Training set metrics:")
        display_metrics(y_train, y_pred_train, mtype)
        print()

        y_pred_test = model.predict(X_test)
        output = pd.DataFrame({'PassengerId': raw_test['PassengerId'], 'Survived': y_pred_test})
        os.makedirs('submissions', exist_ok=True)
        output.to_csv(f'submissions/{mtype}_submission.csv', index=False)

if __name__ == '__main__':
    main()
