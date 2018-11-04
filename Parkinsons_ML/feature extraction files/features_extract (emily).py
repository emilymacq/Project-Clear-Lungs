import pandas as pd
import numpy as np
import tsfresh as ts
import csv

col_headers = ['Time_sec', 'Sens_L1', 'Sens_L2', 'Sens_L3', 'Sens_L4', 'Sens_L5', 'Sens_L6', 'Sens_L7', 'Sens_L8',
            'Sens_R1', 'Sens_R2', 'Sens_R3', 'Sens_R4', 'Sens_R5', 'Sens_R6', 'Sens_R7', 'Sens_R8', 'TF_L', 'TF_R']

# Load demographics to find which patients data need to be downloaded
directory = '/Users/EmilyMac/Documents/MHeal/GitHub/Project-Clear-Lungs/Parkinsons_ML/input/'
readfile = 'GaPt07_01.txt'
demographic_data = pd.read_csv("/Users/EmilyMac/Documents/MHeal/GitHub/Project-Clear-Lungs/Parkinsons_ML/input/demographics.txt", delim_whitespace=True, na_values='NaN')
sample_data = pd.read_csv(directory + readfile, delim_whitespace=True, na_values='NaN', header=None, names=col_headers)

left_ft = sample_data.iloc[:,[0,17]]
right_ft = sample_data.ix[:,[0,18]]
y_exact = demographic_data['HoehnYahr']


features = []
feature_names = ['abs_energy_L', 'abs_energy_R', 'kurtosis_L', 'kurtosis_R', 'skewness_L', 'skewness_R', 'variance_larger_than_standard_deviation_L', 'variance_larger_than_standard_deviation_R', 'cid_ce_L', 
'cid_ce_R', 'count_above_mean_L', 'count_above_mean_R', 'count_below_mean_L', 'count_below_mean_R', 'energy_ratio_by_chunks_L', 'energy_ratio_by_chunks_R', 'label']
filedata = []

features.append(ts.feature_extraction.feature_calculators.abs_energy(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.abs_energy(right_ft['TF_R']))
features.append(ts.feature_extraction.feature_calculators.kurtosis(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.kurtosis(right_ft['TF_R']))
features.append(ts.feature_extraction.feature_calculators.skewness(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.skewness(right_ft['TF_R']))
features.append(ts.feature_extraction.feature_calculators.variance_larger_than_standard_deviation(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.variance_larger_than_standard_deviation(right_ft['TF_R']))
features.append(ts.feature_extraction.feature_calculators.cid_ce(left_ft['TF_L'], True))
features.append(ts.feature_extraction.feature_calculators.cid_ce(right_ft['TF_R'], True))
features.append(ts.feature_extraction.feature_calculators.count_above_mean(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.count_above_mean(right_ft['TF_R']))
features.append(ts.feature_extraction.feature_calculators.count_below_mean(left_ft['TF_L']))
features.append(ts.feature_extraction.feature_calculators.count_below_mean(right_ft['TF_R']))
left_energy = ts.feature_extraction.feature_calculators.energy_ratio_by_chunks(left_ft['TF_L'], [{'num_segments': 5, 'segment_focus': 0}, {'num_segments': 5, 'segment_focus': 1}, {'num_segments': 5, 'segment_focus': 2}, {'num_segments': 5, 'segment_focus': 3}, {'num_segments': 5, 'segment_focus': 4}])
i = 1
for elem in left_energy:
	features.append(elem[1])
	feature_names.append('energy_ratio_by_chunks_' + str(i) + '_L')
	i += 1
right_energy = ts.feature_extraction.feature_calculators.energy_ratio_by_chunks(right_ft['TF_R'], [{'num_segments': 5, 'segment_focus': 0}, {'num_segments': 5, 'segment_focus': 1}, {'num_segments': 5, 'segment_focus': 2}, {'num_segments': 5, 'segment_focus': 3}, {'num_segments': 5, 'segment_focus': 4}])
i = 1
for elem in right_energy:
	features.append(elem[1])
	feature_names.append('energy_ratio_by_chunks_' + str(i) + '_R')
	i += 1
# Check if sample patient has Parkinson's
label = readfile[2:4]
if label == "Pt":
	features.append(1)
else:
	features.append(-1)


filedata.append(features)

# Write data to CSV file
writefile = open('Sample Features.csv', 'w')
with writefile:
	writer = csv.writer(writefile)
	writer.writerows(filedata)

print('Writing complete')







#for i in range(len(feature_names)):
#	print(feature_names[i], features_left[i], features_right[i])

