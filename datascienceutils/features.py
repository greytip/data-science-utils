def extractFeaturesTimeSeries(timeSeries, idCol, timeCol):
    from tsfresh import extract_relevant_features
    return extract_relevant_features(timeSeries, column_id=idCol, column_sort=timeCol)
