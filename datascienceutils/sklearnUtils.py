import copy
import fnmatch
import numpy as np
import os
import pandas as pd
import json

from collections import defaultdict
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer


from . import settings

def feature_scale_or_normalize(dataframe, col_names, norm_type='StandardScaler'):
    """
    Basically converts floating point or integer valued columns to fit into the range of 0 to 1
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    if norm_type=='StandardScaler':
        return StandardScaler().fit_transform(dataframe[col_names])
    elif norm_type=='MinMaxScaler':
        return MinMaxScaler().fit_transform(dataframe[col_names])
    else:
        return None

def feature_select(dataframe, target=None, selector='variance', **kwargs):
    from sklearn.feature_selection import VarianceThreshold, SelectKBest
    if selector=='variance':
        selector = VarianceThreshold()
        return selector.fit_transform(dataframe)
    elif selector == 'SelectKBest':
        assert target
        return SelectKBest(chi2, k=2).fit_transform(dataframe, target)
    else:
        raise "Don't know this feature selector"

def feature_standardize(dataframe, col_names):
    """
    In essence makes sure the column values obey-or-very close to the standard-z- distribution
    But how?? and why is it not confounding all yours to think before using this, but
    generally this helps logistic regression models from being domminated by high variance variables.
    From here: http://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
    """
    from sklearn.preprocessing import scale
    return scale(dataframe[col_names])

def binarize_labels(dataframe, column):
    if dataframe[column].nunique() == 2:
        enc = LabelBinarizer()
        binarized_labels = enc.fit_transform([dataframe[column].tolist(), (dataframe[column].nunique(),)])
    else:
        # Ugh.. I just can't understand how this class is helpful.Rolling my own
        labeled_samples = pd.factorize(dataframe[column])
        #enc = MultiLabelBinarizer()
        binarized_labels = list()
        for each in labeled_samples[0]:
            tmp = [0 for i in range(dataframe[column].nunique())]
            tmp[each] = each
            binarized_labels.append(tmp)
        binarized_labels = np.asarray(binarized_labels)
    return binarized_labels

def dump_model(model, filename, model_params):
    """
    @params:
        @model: actual scikits-learn (or supported by sklearn.joblib) model file
        @model_params: parameters used to build the model
        @filename: Filename to store the model as.

    @output:
        Dumps the model and the parameters as separate files
    """
    import uuid
    from sklearn.externals import joblib

    assert model, "Model required"
    assert filename, "Filename Required"
    assert model_params, "model parameters (dict required)"
    assert model_params['model_type'], "model_type required in model_params"

    model_params.update({'filename': filename,
                         'id': str(uuid.uuid4())})

    with open(os.path.join(settings.MODELS_BASE_PATH,
                            model_params['id'] + '_params_' + filename + '.json'),
                      'w') as params_file:
        json.dump(model_params, params_file)

    joblib.dump(model, os.path.join(settings.MODELS_BASE_PATH,
                                    model_params['id'] + '_' + filename + '.pkl'), compress=('lzma', 3))

def load_model(filename, model_type=None):
    """
    @params:
        @filename: Filename..
        @model_type: Pass, if you can't find filename. we use regex on the settings.models_base_path to find matching
                filenames and pick latest file for the model

    @return:
        @model: joblib.load(filename). basically the model
        @params: The parameters the model  was stored with  if it was
    """
    foldername = settings.MODELS_BASE_PATH
    if not filename:
        assert model_type, 'model_type or filename mandatory'
        relevant_models = list(filter(lambda x: fnmatch.fnmatch(x, '*' + model_type + '*.pkl'), os.listdir(foldername)))
        assert relevant_models, "no relevant models found"
        relevant_models.sort(key=lambda x: os.stat(os.path.join(foldername, x)).st_mtime, reverse=True)
        model = joblib.load(os.path.join(foldername, relevant_models[0]))
        names = elevant_models[0].split('_')
    else:
        model = joblib.load(os.path.join(foldername, filename))
        names = filename.split('_')
    names.insert(1, 'params')
    with open(os.path.join(foldername, '_'.join(names)) + '.json', 'r') as fd:
        params = json.load(fd)
    return model, params

def load_latest_model(foldername, modelType='knn'):
    """
    Parses through the files in the model folder and returns the latest model
    @modelType: can be overloaded to match any string. though the function surrounds a * after value
    """
    assert foldername, "Please pass in a foldername"
    relevant_models = list(filter(lambda x: fnmatch.fnmatch(x, '*' + modelType + '*.pkl'), os.listdir(foldername)))
    assert relevant_models, "no relevant models found"
    relevant_models.sort(key=lambda x: os.stat(os.path.join(foldername, x)).st_mtime, reverse=True)
    latest_model = relevant_models[0]
    return joblib.load(os.path.join(foldername,latest_model))

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.encoders = defaultdict(LabelEncoder)
        if columns:
            self.columns = columns # array of column names to encode
            for each in self.columns:
                self.encoders[each] = LabelEncoder()

    def fit(self,X,y=None):
        return self # not relevant here

    def reverse_transform_all(self, X):
        output = copy.deepcopy(X)
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.encoders[col].inverse_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = self.encoders[col].inverse_transform(col)
        return output

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = copy.deepcopy(X)
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.encoders[col].fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = self.encoders[col].fit_transform(col)
        return output

    def reverse_transform(self, X, y=None):
        return self.fit(X,y).inverse_transform(X)

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)



def cross_val_predict_score(model, actuals, predictions):
    from sklearn.cross_validation import cross_val_predict
    return cross_val_predict(model, actuals, predictions)

def accuracy_calc(actuals, predictions):
    from sklearn.metrics import accuracy_score
    return accuracy_score(actuals, predictions)
