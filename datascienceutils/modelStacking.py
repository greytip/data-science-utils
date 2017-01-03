from .utils import *
def trainVotingClassifier(dataframe, target, **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    clf1 = get_model_obj('knn', random_state=123)
    clf2 = get_model_obj('randomForest', random_state=123)
    clf3 = get_model_obj('gaussianNB', **kwargs)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                                                    voting='soft',
                                                    weights=[1, 1, 5])
    return eclf


def predictVotingClassify(model, dataframe):
    # predict class probabilities for all classifiers
    probas = [c.fit(dataframe, target).predict_proba(dataframe) for c in model.classifiers]
                                #in (clf1, clf2, clf3, eclf)]
