#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
import sys
import traceback
import warnings

script_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(script_dir, 'input')
train_path = os.path.join(input_dir, 'train.csv')
test_path = os.path.join(input_dir, 'test.csv')
output_dir = os.path.join(script_dir, 'output')
output_path = lambda model: os.path.join(output_dir, f'{model}_submission.csv')

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def preprocess(raw_data, is_train):
    # These variables represent info from the training set that needs to be preserved
    global cat_attribs, num_attribs, scaler_X, scaler_y, drop_missing, drop_corr, final_attribs

    df = raw_data
    n_instances = df.shape[0]

    # Remove missing data

    if is_train:
        # Delete the columns with missing data
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data = missing_data[missing_data['Total'] > 0]

        drop_missing = list((missing_data[missing_data['Total'] > 1]).index)
        df = df.drop(drop_missing, axis=1)
        df = df.drop(df.loc[df['Electrical'].isnull()].index)
        assert(df.isnull().sum().max() == 0)
    else:
        # Drop the columns that had missing data in the training set
        df = df.drop(drop_missing, axis=1)
        # Fill in missing values with the mean
        df = df.fillna(df.mean())

    # Remove outliers

    if is_train:
        df = df.drop(df[df['Id'] == 1299].index)
        df = df.drop(df[df['Id'] == 524].index)

    # Make sure SalePrice, GrLivArea, and TotalBsmtSF are normally distributed to satisfy some assumptions of the ML algorithms

    if is_train:
        df['SalePrice'] = np.log(df['SalePrice'])
    df['GrLivArea'] = np.log(df['GrLivArea'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # 0 passed to log function
        df.loc[df['TotalBsmtSF'] > 0, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

    # Standardize numerical attributes

    if is_train:
        cat_attribs = list(df.dtypes[df.dtypes == object].index) + ['MSSubClass']
        num_attribs = list(set(df.columns).difference(cat_attribs))
        num_attribs.remove('Id')
        num_attribs.remove('SalePrice')

        scaler_X = StandardScaler()
        df[num_attribs] = scaler_X.fit_transform(df[num_attribs])

        scaler_y = StandardScaler()
        df['SalePrice'] = scaler_y.fit_transform(df['SalePrice'].values.reshape((-1, 1)))
    else:
        df[num_attribs] = scaler_X.transform(df[num_attribs])

    # One-hot encode categorical features

    df = pd.get_dummies(df, columns=cat_attribs)

    if is_train:
        # Drop features that are highly correlated

        drop_corr = ['Id']
        corrmat = df.corr()
        for colno, col in enumerate(corrmat.columns):
            for rowno, value in enumerate(corrmat[col]):
                if np.abs(value) > 0.8:
                    row = corrmat.index[rowno]
                    if rowno < colno and 'SalePrice' not in [row, col]:
                        drop_corr.append(row)

        df = df.drop(drop_corr, axis=1)

        final_attribs = list(df.columns)
        final_attribs.remove('SalePrice')
    else:
        # Make sure we have the same features as the training set

        need_attribs = set(final_attribs).difference(df.columns)
        dont_need_attribs = set(df.columns).difference(final_attribs)

        #print("[test set] Adding one-hot features:", *need_attribs)
        for attrib in need_attribs:
            # These correspond to categorical features that were present in the training set but not the test set
            df[attrib] = pd.Series(0, index=range(n_instances))
        
        #print("[test set] Dropping one-hot features:", *dont_need_attribs)
        df = df.drop(list(dont_need_attribs), axis=1) # vice versa

        assert(set(df.columns) == set(final_attribs))
        df = df[final_attribs] # Make sure the ordering of the columns is the same as it was in the training set

    return df

def display_metrics(y_true, y_pred, set_desc, model_desc):
    print(f"{set_desc} metrics for {model_desc}:")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE:", rmse)

    y_train_std = scaler_y.scale_[0]
    factor = np.exp(y_train_std * rmse)
    print(f"On average, the regressor is off by a factor of {factor}")

    print()

def main():
    warnings.showwarning = warn_with_traceback
    mtype = sys.argv[1] if len(sys.argv) > 1 else 'all'
    cv = 'cv' in sys.argv[1:]

    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    df_train = preprocess(raw_train, is_train=True)
    df_test = preprocess(raw_test, is_train=False)

    X_train, y_train = df_train.drop('SalePrice', axis=1), df_train['SalePrice']
    X_test = df_test

    model_types = ['svr', 'sgd', 'ridge', 'lasso'] if mtype == 'all' else [mtype]
    for model_type in model_types:
        #model = LinearRegression() if model_type == 'linear' else \
        model = LinearSVR(dual=False, loss='squared_epsilon_insensitive') if model_type == 'svr' else \
                SGDRegressor(tol=1e-6, random_state=42) if model_type == 'sgd' else \
                Ridge(alpha=1e-10) if model_type == 'ridge' else \
                Lasso(alpha=1e-4) if model_type == 'lasso' else \
                None
        if model is None:
            raise Exception(f'Unknown model: {model_type}')
        
        if not cv:
            model.fit(X_train, y_train)
        else:
            # Perform cross-validation

            scores = []
            candidates = []
            kfold = KFold(10, shuffle=True, random_state=42)

            for i, (tr_ix, val_ix) in enumerate(kfold.split(df_train)):
                candidate = clone(model)
                tr, val = df_train.iloc[tr_ix, :], df_train.iloc[val_ix, :]
                X_tr, y_tr, X_val, y_val = tr.drop('SalePrice', axis=1), tr['SalePrice'], val.drop('SalePrice', axis=1), val['SalePrice']
                candidate.fit(X_tr, y_tr)
                y_pred_val = candidate.predict(X_val)
                score = -np.sqrt(mean_squared_error(y_val, y_pred_val))
                print(f"[cv] Model {i} has RMSE {-score}")

                scores.append(score)
                candidates.append(candidate)
            
            #print(scores)
            best_score = np.max(scores)
            best_model = candidates[np.argmax(scores)]
            print(f"[cv] Best model has RMSE of {-best_score} on the validation set")
            model = best_model

        y_pred_train = model.predict(X_train)
        display_metrics(y_train, y_pred_train, "Training set", model_type)

        y_pred_test = model.predict(X_test)
        y_train_mean, y_train_std = scaler_y.mean_[0], scaler_y.scale_[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            adjusted_y_pred_test = np.exp((y_train_std * y_pred_test) + y_train_mean) # todo: fix overflow warning

        output = pd.DataFrame({'Id': raw_test['Id'], 'SalePrice': adjusted_y_pred_test})
        os.makedirs(output_dir, exist_ok=True)
        output.to_csv(output_path(model_type), index=False, float_format='%.2f')

if __name__ == '__main__':
    main()
