import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    if modelType == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        #n= 6 seems to give the best trade-off between accuracy and precision
        model = KNeighborsClassifier(**kwargs)

    elif modelType == 'gaussianNB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    elif modelType == 'sgd':
        # Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier()

    elif modelType == 'perceptron':
        from sklearn.linear_model import Perceptron
        model = Perceptron()
    elif modelType == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBClassifier(**kwargs)
    else:
        pass

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
    #TODO: Damn I'm sick of looking at this if spaghetti. Next task is to put into a dict and call
    # the function
    if modelType == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        # 6 seems to give the best trade-off between accuracy and precision
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(dataframe, target)
        return knn

    elif modelType == 'gaussianNB':
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(dataframe, target)
        return gnb

    elif modelType == 'multinomialNB':
        from sklearn.naive_bayes import MultinomialNB
        # TODO: figure out how to configure binomial distribution
        mnb = MultinomialNB()
        mnb.fit(dataframe, target)
        return mnb

    elif modelType == 'bernoulliNB':
        from sklearn.naive_bayes import BernoulliNB
        bnb = BernoulliNB()
        bnb.fit(dataframe, target)
        return bnb

    elif modelType == 'randomForest':
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=234, **kwargs)
        rfc.fit(dataframe, target)
        return rfc

    elif modelType == 'svm':
        from sklearn.svm import SVC
        svc = SVC(random_state=0, probability=True)
        svc.fit(dataframe, target)
        return svc

    elif modelType == 'votingClass':
        tVC = trainVotingClassifier(dataframe, target, **kwargs)
        return tVC

    elif modelType == 'linearRegression':
        #assert column, "Column name required for building a linear model"
        #assert dataframe[column].shape == target.shape
        from sklearn import linear_model
        l_reg = linear_model.LinearRegression()
        if column:
            l_reg.fit(dataframe[column].reshape(len(target), 1), target)
        else:
            l_reg.fit(dataframe, target)
        return l_reg

    elif modelType == 'logisticRegression':
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(random_state=123)
        if column:
            log_reg.fit(np.asarray(dataframe[column], dtype='float64').reshape(len(target),1), target)
        else:
            log_reg.fit(dataframe, target)
        return log_reg
    elif modelType == 'kde':
         from sklearn.neighbors.kde import KernelDensity
         kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dataframe)
         return kde

    elif modelType == 'AR':
        import statsmodels.api as sm
        # fit an AR model and forecast
        ar_fitted = sm.tsa.AR(dataframe).fit(maxlag=9, method='mle', disp=-1)
        #ts_forecast = ar_fitted.predict(start='2008', end='2050')
        return ar_fitted

    elif modelType == 'SARIMAX':
        mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
        return mod

    elif modelType == 'sgd':
        # Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier()
        return sgd

    elif modelType == 'perceptron':
        from sklearn.linear_model import Perceptron
        perceptron = Perceptron()
        return perceptron
    elif modelType == 'xgboost':
        import xgboost as xgb
        gbm = xgb.XGBClassifier(**kwargs).fit(dataframe, target)
        return gbm

    else:
        raise ''
        pass



def featureSelect(dataframe):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return sel.fit_transform(dataframe)

if __name__ == '__main__':
    pass
