def featureSelect(dataframe):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return sel.fit_transform(X)

def accuracy_calc(actuals, predictions):
    from sklearn.metrics import accuracy_score
    return accuracy_score(actuals, predictions)

def cross_val_predict_score(model, actuals, predictions):
    from sklearn.cross_validation import cross_val_predict
    return cross_val_predict(model, actuals, predictions)

class MultiColumnLabelEncoder:
    '''
    >>> MultiColumnLabelEncoder(columns = ['appsitecat','adposition','gender','OS','Carrier','DeviceType','country','traffictype'])
    '''
    def __init__(self,columns = None):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = le.fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = le.fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def dump_model(model, filename):
    assert model, "Model required"
    assert filename, "Filename Required"
    from sklearn.externals import joblib
    joblib.dump(model, filename+ '.pkl')

if __name__ == '__main__':
    pass
