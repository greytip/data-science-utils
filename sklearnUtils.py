from sklearn.preprocessing import LabelEncoder, StandardScaler

def normalize(dataframe, norm_type='StandardScalar'):
    return StandardScaler().fit_transform(dataframe)


def dump_model(model, filename):
    assert model, "Model required"
    assert filename, "Filename Required"
    from sklearn.externals import joblib
    joblib.dump(model, filename+ '.pkl')

def load_latest_model(foldername, modelType='knn'):
    """
    Parses through the files in the model folder and returns the latest model
    @modelType: can be overloaded to match any string. though the function appends a * after value
    """
    assert foldername, "Please pass in a foldername"
    import os, fnmatch
    from sklearn.externals import joblib
    relevant_models = list(filter(lambda x: fnmatch.fnmatch(x, '*' + modelType + '*'), os.listdir(foldername)))
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
        output = X.copy()
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
        output = X.copy()
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
