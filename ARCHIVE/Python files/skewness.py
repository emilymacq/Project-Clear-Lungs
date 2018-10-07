# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.skew.html
import scipy

# a: ndarray (data)
# returns skewness (ndarray) - skewness of values along an axis
scipy.stats.skew(a, axis=0, bias=True)