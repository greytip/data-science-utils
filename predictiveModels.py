import matplotlib.pyplot as plt
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


def train(dataframe, target, column=None, modelType='knn', cross_val=False):
    """
    Generic training wrapper around different scikits-learn models

    @params:
        @dataframe: A pandas dataframe with all feature columns.
        @target: pandas series or numpy array(basically a iterable) with the target values. should match length with dataframe
        @modelType: String representing the model you want to train with
        @cross_val: Boolean, if True returns the  model object (without training)for you to perform cross_valuation afterwards

    @return:
        Model object
    """

    if modelType =='knn':
        from sklearn.neighbors import KNeighborsClassifier
        # 6 seems to give the best trade-off between accuracy and precision
        knn = KNeighborsClassifier(n_neighbors=6)
        if cross_val:
            return knn
        knn.fit(dataframe, target)
        return knn

    elif modelType=='gaussianNB':
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        if cross_val:
            return gnb
        gnb.fit(dataframe, target)
        return gnb

    elif modelType=='multinomialNB':
        from sklearn.naive_bayes import MultinomialNB
        # TODO: figure out how to configure binomial distribution
        mnb = MultinomialNB()
        if cross_val:
            return mnb
        mnb.fit(dataframe, target)
        return mnb

    elif modelType=='bernoulliNB':
        from sklearn.naive_bayes import BernoulliNB
        bnb = BernoulliNB()
        if cross_val:
            return bnb
        bnb.fit(dataframe, target)
        return bnb

    elif modelType=='randomForest':
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=234)
        if cross_val:
            return rfc
        rfc.fit(dataframe, target)
        return rfc

    elif modelType == 'votingClass':
        tVC = trainVotingClassifier(dataframe, target)
        return tVC

    elif modelType == 'linearRegression':
        #assert column, "Column name required for building a linear model"
        #assert dataframe[column].shape == target.shape
        from sklearn import linear_model
        l_reg = linear_model.LinearRegression()
        if cross_val:
            return l_reg
        if column:
            l_reg.fit(dataframe[column], target)
        else:
            l_reg.fit(dataframe, target)
        return l_reg

    elif modelType == 'AR':
        import statsmodels.api as sm
        # fit an AR model and forecast
        ar_fitted = sm.tsa.AR(dataframe).fit(maxlag=9, method='mle', disp=-1)
        #ts_forecast = ar_fitted.predict(start='2008', end='2050')
        return ar_fitted

    elif modelType == 'SARIMAX':
        mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
        return mod

    else:
        raise ''
        pass



def featureSelect(dataframe):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return sel.fit_transform(dataframe)

if __name__ == '__main__':
    pass
