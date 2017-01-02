import matplotlib.pyplot as plt
import numpy as np
import os

from collections import defaultdict
from sklearn import model_selection, metrics

# Sigh lightgbm insist this is the only wa
os.environ['LIGHTGBM_EXEC'] = os.path.join(os.getenv("HOME"), 'bin', 'lightgbm')

def get_model_obj(modelType, **kwargs):
    if modelType == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        # 6 seems to give the best trade-off between accuracy and precision
        knn = KNeighborsClassifier(n_neighbors=6, **kwargs)
        return knn
    elif modelType == 'gaussianNB':
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB(**kwargs)
        return gnb

    elif modelType == 'multinomialNB':
        from sklearn.naive_bayes import MultinomialNB
        # TODO: figure out how to configure binomial distribution
        mnb = MultinomialNB(**kwargs)
        return mnb

    elif modelType == 'bernoulliNB':
        from sklearn.naive_bayes import BernoulliNB
        bnb = BernoulliNB(**kwargs)
        return bnb

    elif modelType == 'randomForest':
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=234, **kwargs)
        return rfc

    elif modelType == 'svm':
        from sklearn.svm import SVC
        svc = SVC(random_state=0, probability=True, **kwargs)
        return svc

    elif modelType == 'votingClass':
        tVC = trainVotingClassifier(dataframe, target, **kwargs)
        return tVC

    elif modelType == 'linearRegression':
        #assert column, "Column name required for building a linear model"
        #assert dataframe[column].shape == target.shape
        from sklearn import linear_model
        l_reg = linear_model.LinearRegression(**kwargs)
        return l_reg

    elif modelType == 'logisticRegression':
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(random_state=123, **kwargs)
        return log_reg

    elif modelType == 'kde':
         from sklearn.neighbors.kde import KernelDensity
         kde = KernelDensity(kernel='gaussian', bandwidth=0.2, **kwargs).fit(dataframe)
         return kde

    elif modelType == 'AR':
        import statsmodels.api as sm
        # fit an AR model and forecast
        ar_fitted = sm.tsa.AR(dataframe).fit(maxlag=9, method='mle', disp=-1, **kwargs)
        #ts_forecast = ar_fitted.predict(start='2008', end='2050')
        return ar_fitted

    elif modelType == 'SARIMAX':
        mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0),
                seasonal_order=(1,1,1,12), **kwargs)
        return mod

    elif modelType == 'sgd':
        # Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(**kwargs)
        return sgd

    elif modelType == 'perceptron':
        from sklearn.linear_model import Perceptron
        perceptron = Perceptron(**kwargs)
        return perceptron

    elif modelType == 'xgboost':
        import xgboost as xgb
        xgbm = xgb.XGBClassifier(**kwargs).fit(dataframe, target)
        return xgbm

    elif modelType == 'baseNN':
        from keras.models import Sequential
        from keras.layers import Dense
        # create model
        model = Sequential()
        assert args.get('inputParams', None)
        assert args.get('outputParams', None)
        model.add(Dense(inputParams))
        model.add(Dense(outputParams))
        if args.get('compileParams'):
            # Compile model
            model.compile(compileParams)# loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    elif modelType == 'lightGBMRegression':
        from pylightgbm.models import GBMRegressor
        lgbm_lreg = GBMRegressor(   num_iterations=100, early_stopping_round=10,
                                    num_leaves=10, min_data_in_leaf=10)
        return lgbm_lreg

    elif modelType == 'lightGBMBinaryClass':
        from pylightgbm.models import GBMClassifier
        lgbm_bc = GBMClassifier(metric='binary_error', min_data_in_leaf=1)
        return lgbm_bc

    else:
        raise ''

def trainVotingClassifier(dataframe, target):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                                                    voting='soft',
                                                    weights=[1, 1, 5])
    return eclf

def predictVotingClassify(model, dataframe):
    # predict class probabilities for all classifiers
    probas = [c.fit(dataframe, target).predict_proba(dataframe) for c in model.classifiers]
                                #in (clf1, clf2, clf3, eclf)]


def cross_val_train(dataframe, target, modelType, **kwargs):
    cv = kwargs.pop('cv',None)
    model = get_model_obj(modelType, **kwargs)
    scores = cross_val_score(model, dataframe, target, cv=cv)
    return scores

def train(dataframe, target, modelType, column=None, **kwargs):
    """
    Generic training wrapper around different scikits-learn models

    @params:
        @dataframe: A pandas dataframe with all feature columns.
        @target: pandas series or numpy array(basically a iterable) with the target values. should match length with dataframe
        @modelType: String representing the model you want to train with

    @return:
        Model object
    """
    model = get_model_obj(modelType, **kwargs)
    if column:
        source = dataframe[column].reshape((len(target), 1))
        model.fit(source, target)
    else:
        model.fit(dataframe, target)
    return model


def grid_search(dataframe, target, modelType, **kwargs):
    model = get_model_obj(modelType, **kwargs)
    scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
    clf = model_selection.GridSearchCV(model, scoring=scorer, cv=2)
    clf.fit(dataframe, target)
    return clf

def featureSelect(dataframe):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return sel.fit_transform(dataframe)

if __name__ == '__main__':
    pass
