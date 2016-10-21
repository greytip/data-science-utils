import copy

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def feature_scale_or_normalize(dataframe, col_names, norm_type='StandardScalar'):
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
    labels = dataframe[column].values
    from sklearn import preprocessing
    enc = preprocessing.LabelBinarizer()
    binarized_labels = enc.fit_transform(labels)
    dataframe.drop(column, axis=1, inplace=True)
    return dataframe, binarized_labels

def dump_model(model, filename):
    assert model, "Model required"
    assert filename, "Filename Required"
    from sklearn.externals import joblib
    joblib.dump(model, filename+ '.pkl')

def load_latest_model(foldername, modelType='knn'):
    """
    Parses through the files in the model folder and returns the latest model
    @modelType: can be overloaded to match any string. though the function surrounds a * after value
    """
    assert foldername, "Please pass in a foldername"
    import os, fnmatch
    from sklearn.externals import joblib
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
