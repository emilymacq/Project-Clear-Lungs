"""
This script is written to do analysis on GA study
"""

# import libraries
import pandas as pd
import numpy as np
import tsfresh
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# import methods from other scripts / packages
from feature_extraction.FFT import fft_extractor
from feature_extraction.abs_energy import AbsoluteEnergyCalculator

# Constant declarations
col_names = ['Time_sec', 'Sens_L1', 'Sens_L2', 'Sens_L3', 'Sens_L4', 'Sens_L5', 'Sens_L6', 'Sens_L7', 'Sens_L8',
             'Sens_R1', 'Sens_R2', 'Sens_R3', 'Sens_R4', 'Sens_R5', 'Sens_R6', 'Sens_R7', 'Sens_R8', 'TF_L', 'TF_R']

"""
Main Controller
"""
def study_ga_controller(demographics_data):
    group_1_data, group_2_data = split_group_data(demographics_data)
    group_1_analysis(group_1_data)
    # group_2_analysis(group_2_data)


def split_group_data(demographics_data):
    group_1_data = demographics_data[demographics_data['Group'] == 1]
    group_2_data = demographics_data[demographics_data['Group'] == 2]
    return group_1_data, group_2_data


def print_newline():
    print("")


def print_seperator():
    print("--------------------------")


# ----------------------------------------------------- GROUP 1 ------------------------------------------------------ #


def group_1_analysis(group_1_data):
    print_newline()
    print("#####################################")
    print("Group 1 Analysis:")
    print("#####################################")

    # Read data from a participant in group 1 - PD Patient
    # GaPt14 : Age 56 Gender 1
    gapt14_data_1 = pd.read_csv("../input/GaPt14_01.txt", header=None, sep='\t', names=col_names)
    # gapt14_data_2 = pd.read_csv("../input/GaPt14_02.txt", header=None, sep='\t', names=col_names)
    # gapt14_data_10 = pd.read_csv("../input/GaPt14_10.txt", header=None, sep='\t', names=col_names)

    # GaPt21 : Age 81 Gender 1
    gapt21_data_1 = pd.read_csv("../input/GaPt21_01.txt", header=None, sep='\t', names=col_names)
    # gapt21_data_2 = pd.read_csv("../input/GaPt21_02.txt", header=None, sep='\t', names=col_names)
    # gapt21_data_10 = pd.read_csv("../input/GaPt21_10.txt", header=None, sep='\t', names=col_names)

    # GaPt07 : Age 57 Gender 2
    # gapt7_data_1 = pd.read_csv("../input/GaPt07_01.txt", header=None, sep='\t', names=col_names)
    # gapt7_data_2 = pd.read_csv("../input/GaPt07_02.txt", header=None, sep='\t', names=col_names)

    # GaPt26 : Age 78 Gender 2
    # gapt26_data_1 = pd.read_csv("../input/GaPt26_01.txt", header=None, sep='\t', names=col_names)
    # gapt26_data_2 = pd.read_csv("../input/GaPt26_02.txt", header=None, sep='\t', names=col_names)
    # gapt26_data_10 = pd.read_csv("../input/GaPt26_10.txt", header=None, sep='\t', names=col_names)


    # plot_patient_data(gapt14_data_1, 'Time_sec', 'TF_L', "Total force on left foot for patient: Group 1 GaPt14")
    # plot_zoomed_patient_data(gapt14_data_1, 'Time_sec', 'TF_L', "Total force on left foot for patient: Group 1 GaPt14")

    all_patient_dataframe = pd.DataFrame(columns=['Patient_Number', 'Study', 'Foot', 'Median', 'Max', 'Min', 'Skewness', 'Std', 'Variance', 'Abs_Energy',
                                              'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                                              'aggtype_centroid', 'aggtype_variance', 'aggtype_skew', 'aggtype_kurtosis'])

    df1 = pd.DataFrame([[np.nan] * len(all_patient_dataframe.columns)], columns=all_patient_dataframe.columns)

    all_patient_dataframe = df1.append(all_patient_dataframe, ignore_index=True)
    all_patient_dataframe = add_patient_data(all_patient_dataframe, gapt14_data_1, 14, 'GaPt', 'left')
    all_patient_dataframe = df1.append(all_patient_dataframe, ignore_index=True)
    all_patient_dataframe = add_patient_data(all_patient_dataframe, gapt14_data_1, 14, 'GaPt', 'right')

    all_patient_dataframe = df1.append(all_patient_dataframe, ignore_index=True)
    add_patient_data(all_patient_dataframe, gapt21_data_1, 21, 'GaPt')



def add_patient_data(all_patient_dataframe, patient_data, patient_number, patient_study, foot):
    all_patient_dataframe['Patient_Number'] = patient_number
    all_patient_dataframe['Study'] = patient_study
    all_patient_dataframe['Foot'] = foot

    patient_row_entry = all_patient_dataframe[(all_patient_dataframe['Patient_Number'] == patient_number) & (all_patient_dataframe['Study'] == patient_study) & (all_patient_dataframe['Foot'] == foot)]
    patient_row_entry = extract_features(patient_row_entry, patient_data)
    return all_patient_dataframe


# ----------------------------------------------------- GROUP 2 ------------------------------------------------------ #

def group_2_analysis(group_2_data):
    print_newline()
    print("#####################################")
    print("Group 2 Analysis:")
    print("#####################################")

    # group_2_study_ga()
    # group_2_study_ju()
    # group_2_study_si()


# -------------------------------------------------- PLOTTING METHODS ------------------------------------------------ #

def find_gait_cycle(patient_data):
    '''
    gait_cycle = pd.DataFrame(patient_data[(patient_data['Sens_L1'] == 0) & (patient_data['Sens_L2'] == 0) &
                                           (patient_data['Sens_L3'] == 0) & (patient_data['Sens_L4'] == 0) &
                                           (patient_data['Sens_L5'] == 0) & (patient_data['Sens_L6'] == 0) &
                                           (patient_data['Sens_L7'] == 0) & (patient_data['Sens_L8'] == 0)]['Time_sec'])
    '''
    gait_cycle = pd.DataFrame(patient_data[(patient_data['TF_L'] == 0)]['Time_sec'])
    gait_cycle['Time_sec'] = gait_cycle['Time_sec'].astype(int)
    gait_cycle = gait_cycle['Time_sec'].unique()

    print_newline()
    print_seperator()
    print("Values with zero VGRF:\n")
    print(gait_cycle)
    print_seperator()


def extract_features(patient_entry, patient_data):

    fft = fft_extractor()
    patient_entry = add_foot_coeffs(patient_entry, fft, patient_data, 'left')
    patient_entry = add_foot_coeffs(patient_entry, fft, patient_data, 'right')

    abs_en = AbsoluteEnergyCalculator()
    patient_entry['Abs_Energy'] = abs_en.calculate_abs_energy(patient_data[['Time_sec', 'TF_L']], 'TF_L')
    # extract_eda_features(patient_data[['Time_sec', 'TF_L']], 'TF_L')
    return patient_entry

def add_foot_coeffs(patient_entry, fft, patient_data, feet_type):
    if feet_type == 'left':
        foot_coeff = fft.calculate_fft_coeff(patient_data[['Time_sec', 'TF_L']], 'TF_L')
    elif feet_type == 'right':
        foot_coeff = fft.calculate_fft_coeff(patient_data[['Time_sec', 'TF_R']], 'TF_R')
    else:
        raise ValueError("add_foot_coeffs() : Wrong value supplied")

    patient_entry['coeff_1'] = foot_coeff['coeff_1__attr_"real"']
    patient_entry['coeff_2'] = foot_coeff['coeff_2__attr_"real"']
    patient_entry['coeff_3'] = foot_coeff['coeff_3__attr_"real"']
    patient_entry['coeff_4'] = foot_coeff['coeff_4__attr_"real"']
    return patient_entry


def extract_eda_features(patient_data, col_name):
    feat_median = tsfresh.feature_extraction.feature_calculators.median(patient_data[col_name])
    feat_max = tsfresh.feature_extraction.feature_calculators.maximum(patient_data[col_name])
    feat_min = tsfresh.feature_extraction.feature_calculators.minimum(patient_data[col_name])
    feat_skewness = tsfresh.feature_extraction.feature_calculators.skewness(patient_data[col_name])
    feat_std = tsfresh.feature_extraction.feature_calculators.standard_deviation(patient_data[col_name])
    feat_var = tsfresh.feature_extraction.feature_calculators.variance(patient_data[col_name])
    return


# -------------------------------------------------- PLOTTING METHODS ------------------------------------------------ #

def plot_patient_data(patient_df, x_col_name, y_col_name, plot_title):
    ax = sns.lineplot(x=x_col_name, y=y_col_name, data=patient_df)
    ax.set_title(plot_title)
    plt.show()


def plot_zoomed_patient_data(patient_df, x_col_name, y_col_name, plot_title):
    zoomed_time_data = patient_df[patient_df[x_col_name] < 20]
    ax = sns.lineplot(x=x_col_name, y=y_col_name, data=zoomed_time_data)
    ax.set_title(plot_title)
    plt.show()


def plot_sensor_data(patient_df, x_col_name, y_col_name, sensor_name):
    ax = sns.lineplot(x=x_col_name, y=y_col_name, data=patient_df)
    ax.set_title(sensor_name + "reading over time")
    plt.show()
