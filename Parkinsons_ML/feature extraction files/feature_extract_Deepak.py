import os 
import pandas as pd
import matplotlib.pyplot as plt
import tsfresh.feature_extraction.feature_calculators as tf
from tsfresh import extract_features


def main():
    dirname = os.path.realpath('.')
    filename = dirname + '\\GaPt07_01.txt'

    data = open(filename, 'r')

    totalData = {}
    
    time = []
    totalForceL = []
    totalForceR = []
    id = []
    for line in data:
        tempForce = line.split()
        id.append(1)
        time.append(float(tempForce[0]))
        totalForceL.append(float(tempForce[17]))
        totalForceR.append(float(tempForce[18]))

    totalData["id"] = id
    totalData["time"] = time
    totalData["totalForceL"] = totalForceL
    totalData["totalForceR"] = totalForceR

    dataPandas = pd.DataFrame.from_dict(totalData)

    extracted_features = {}

    #extract_featuresL = extract_features(dataPandas, column_id="id", column_kind=None, column_value=None)
    extracted_features["absEnergyL"] = tf.abs_energy(totalData["totalForceL"])
    extracted_features["absEnergyR"] = tf.abs_energy(totalData["totalForceR"])
    extracted_features["kurtosisL"] = tf.kurtosis(totalData["totalForceL"])
    extracted_features["kurtosisR"] = tf.kurtosis(totalData["totalForceR"])
    extracted_features["skewnessL"] = tf.skewness(totalData["totalForceL"])
    extracted_features["skewnessR"] = tf.skewness(totalData["totalForceR"])
    extracted_features["medianL"] = tf.median(totalData["totalForceL"])
    extracted_features["medianR"] = tf.median(totalData["totalForceR"])
    extracted_features["meanL"] = tf.mean(totalData["totalForceL"])
    extracted_features["meanR"] = tf.mean(totalData["totalForceR"])
    extracted_features["varianceL"] = tf.variance(totalData["totalForceL"])
    extracted_features["varianceR"] = tf.variance(totalData["totalForceR"])

    temp = tf.fft_aggregated(totalData["totalForceL"], [{"aggtype": "centroid"}, {"aggtype": "variance"}, {"aggtype":"skew"}, {"aggtype":"kurtosis"}])
    int = 0
    for list in temp:
        if int == 0:
            extracted_features["fftCentroidL"] = list
        if int == 1:
            extracted_features["fftVarianceL"] = list
        if int == 2:
            extracted_features["fftSkewL"] = list
        if int == 3:
            extracted_features["fftKurtosisL"] = list
        int += 1
    
    temp2 = tf.fft_aggregated(totalData["totalForceR"], [{"aggtype": "centroid"}, {"aggtype": "variance"}, {"aggtype":"skew"}, {"aggtype":"kurtosis"}])
    int = 0
    for list in temp2:
        if int == 0:
            extracted_features["fftCentroidR"] = list
        if int == 1:
            extracted_features["fftVarianceR"] = list
        if int == 2:
            extracted_features["fftSkewR"] = list
        if int == 3:
            extracted_features["fftKurtosisR"] = list
        int += 1




    
if __name__ == "__main__":
    main()