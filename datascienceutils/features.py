from sklearn.feature_selection import VarianceThreshold

#TODO: filter of categorical features by how much of the dataset they divide the records into.
#TODO: filter of numerical features by variance of the features.

def extractFeaturesTimeSeries(timeSeries, idCol, timeCol):
    from tsfresh import extract_relevant_features
    return extract_relevant_features(timeSeries, column_id=idCol, column_sort=timeCol)


