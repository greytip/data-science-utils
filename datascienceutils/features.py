from sklearn.feature_selection import VarianceThreshold
#TODO: https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
#TODO: filter of categorical features by how much of the dataset they divide the records into.
#TODO: filter of numerical features by variance of the features.

def extractFeaturesTimeSeries(timeSeries, idCol, timeCol):
    from tsfresh import extract_relevant_features
    return extract_relevant_features(timeSeries, column_id=idCol, column_sort=timeCol)


