import pandas as pd
import numpy as np
import tsfresh as ts

col_headers = ['Time_sec', 'Sens_L1', 'Sens_L2', 'Sens_L3', 'Sens_L4', 'Sens_L5', 'Sens_L6', 'Sens_L7', 'Sens_L8',
            'Sens_R1', 'Sens_R2', 'Sens_R3', 'Sens_R4', 'Sens_R5', 'Sens_R6', 'Sens_R7', 'Sens_R8', 'TF_L', 'TF_R']

# Load demographics to find which patients data need to be downloaded
demographic_data = pd.read_csv("/Users/EmilyMac/Documents/MHeal/GitHub/Project-Clear-Lungs/Parkinsons_ML/input/demographics.txt", delim_whitespace=True, na_values='NaN')
sample_data = pd.read_csv("/Users/EmilyMac/Documents/MHeal/GitHub/Project-Clear-Lungs/Parkinsons_ML/input/GaPt07_01.txt", delim_whitespace=True, na_values='NaN', header=None, names=col_headers)

left_ft = sample_data.iloc[:,[0,17]]
right_ft = sample_data.ix[:,[0,18]]
y_exact = demographic_data['HoehnYahr']


#features_filtered_direct = ts.extract_relevant_features(left_ft, y_exact,
#                                                     column_id='TF_L', column_sort='Time_sec')
#extracted_features = ts.extract_features(left_ft, column_id='TF_L', column_sort='Time_sec')

features_left = []
feature_names = ['abs_energy', 'kurtosis', 'skewness', 'variance_larger_than_standard_deviation', 'cid_ce', 
'count_above_mean', 'count_below_mean', 'energy_ratio_by_chunks']

features_left.append(ts.feature_extraction.feature_calculators.abs_energy(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.kurtosis(left_ft['TF_L']))
# features.append(ts.feature_extraction.feature_calculators.fft_aggregated(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.skewness(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.variance_larger_than_standard_deviation(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.cid_ce(left_ft['TF_L'], True))
features_left.append(ts.feature_extraction.feature_calculators.count_above_mean(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.count_below_mean(left_ft['TF_L']))
features_left.append(ts.feature_extraction.feature_calculators.energy_ratio_by_chunks(left_ft['TF_L'], [{'num_segments': 5, 'segment_focus': 0}, {'num_segments': 5, 'segment_focus': 1}, {'num_segments': 5, 'segment_focus': 2}, {'num_segments': 5, 'segment_focus': 3}, {'num_segments': 5, 'segment_focus': 4}]))

features_right = []

features_right.append(ts.feature_extraction.feature_calculators.abs_energy(right_ft['TF_R']))
features_right.append(ts.feature_extraction.feature_calculators.kurtosis(right_ft['TF_R']))
# features.append(ts.feature_extraction.feature_calculators.fft_aggregated(left_ft['TF_L']))
features_right.append(ts.feature_extraction.feature_calculators.skewness(right_ft['TF_R']))
features_right.append(ts.feature_extraction.feature_calculators.variance_larger_than_standard_deviation(right_ft['TF_R']))
features_right.append(ts.feature_extraction.feature_calculators.cid_ce(right_ft['TF_R'], True))
features_right.append(ts.feature_extraction.feature_calculators.count_above_mean(right_ft['TF_R']))
features_right.append(ts.feature_extraction.feature_calculators.count_below_mean(right_ft['TF_R']))
features_right.append(ts.feature_extraction.feature_calculators.energy_ratio_by_chunks(right_ft['TF_R'], [{'num_segments': 5, 'segment_focus': 0}, {'num_segments': 5, 'segment_focus': 1}, {'num_segments': 5, 'segment_focus': 2}, {'num_segments': 5, 'segment_focus': 3}, {'num_segments': 5, 'segment_focus': 4}]))



for i in range(len(feature_names)):
	print(feature_names[i], features_left[i], features_right[i])

