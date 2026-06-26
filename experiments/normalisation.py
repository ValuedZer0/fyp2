from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def minmax_scale(X):
    return MinMaxScaler().fit_transform(X)

def standard_scale(X):
    return StandardScaler().fit_transform(X)

def robust_scale(X):
    return RobustScaler().fit_transform(X)