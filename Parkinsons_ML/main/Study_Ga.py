"""
This script is written to do analysis on GA study
"""

# import libraries
import re
import tsfresh
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


# import methods from other scripts / packages
from storage.DataLoader import data_loader
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

    # Create Empty Dataframe
    all_patient_dataframe = pd.DataFrame(
        columns=['Patient_Number', 'Study', 'Foot', 'Median', 'Max', 'Min', 'Skewness', 'Std', 'Variance', 'Abs_Energy',
                 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                 'aggtype_centroid', 'aggtype_variance', 'aggtype_skew', 'aggtype_kurtosis'])

    df1 = pd.DataFrame([[np.nan] * len(all_patient_dataframe.columns)], columns=all_patient_dataframe.columns)

    patient_data_loader = data_loader()
    patient_data_file_paths = patient_data_loader.get_patient_file_paths()

    all_patient_dataframe = GenerateAllPatientDataframe(patient_data_loader, patient_data_file_paths, all_patient_dataframe, df1)

    train_set, test_set = train_test_split(all_patient_dataframe, test_size=0.25, random_state=42)
    
    print("Train Set:")
    print_newline()
    print(train_set)

    print_newline()
    
    print("Test Set:")
    print_newline()
    print(test_set)


def GenerateAllPatientDataframe(patient_data_loader, patient_data_file_paths, all_patient_dataframe, df1):

    for patient_file_path in patient_data_file_paths:
        # Read patient data
        patient_data = patient_data_loader.read_patient_data(patient_file_path)

        filename_fields = extract_fields_from_filename(patient_data_loader, patient_file_path)
        study_name = filename_fields.group(1)
        patient_number = filename_fields.group(2)
        data_file_number = filename_fields.group(3)

        # plot_patient_data(patient_data, 'Time_sec', 'TF_L', "Total force on left foot for patient: Group 1 " + study_name + patient_number)
        # plot_zoomed_patient_data(patient_data, 'Time_sec', 'TF_L', "Total force on left foot for patient: Group 1 " + study_name + patient_number)

        # add empty row entry
        all_patient_dataframe = df1.append(all_patient_dataframe, ignore_index=True)
        all_patient_dataframe = add_patient_data(all_patient_dataframe, patient_data, patient_number, study_name, 'left')

        all_patient_dataframe = df1.append(all_patient_dataframe, ignore_index=True)
        all_patient_dataframe = add_patient_data(all_patient_dataframe, patient_data, patient_number, study_name, 'right')

    return all_patient_dataframe


def extract_fields_from_filename(patient_data_loader, patient_file_path):
    patient_filename = patient_data_loader.extract_file_name(patient_file_path)

    pattern = "([A-Za-z]+)([\d]+)_([\d]+)"
    fields_from_filename = re.match(pattern, patient_filename)
    return fields_from_filename


def add_patient_data(all_patient_dataframe, patient_data, patient_number, patient_study, foot):
    all_patient_dataframe.loc[0, 'Patient_Number'] = patient_number
    all_patient_dataframe.loc[0, 'Study'] = patient_study
    all_patient_dataframe.loc[0, 'Foot'] = foot

    fft = fft_extractor()
    abs_en = AbsoluteEnergyCalculator()
    all_patient_dataframe = extract_features(all_patient_dataframe, patient_data, foot, fft, abs_en)
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


# ------------------------------------------ FEATURE EXTREACTION METHODS --------------------------------------------- #

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


def extract_features(all_patient_dataframe, patient_data, foot, fft, abs_en):
    if foot == "left":
        all_patient_dataframe = add_foot_coeffs(all_patient_dataframe, fft, patient_data, 'left')
        all_patient_dataframe.loc[0, 'Abs_Energy'] = abs_en.calculate_abs_energy(patient_data[['Time_sec', 'TF_L']], 'TF_L')
        all_patient_dataframe = extract_eda_features(patient_data[['Time_sec', 'TF_L']], 'TF_L', all_patient_dataframe)
    elif foot == "right":
        all_patient_dataframe = add_foot_coeffs(all_patient_dataframe, fft, patient_data, 'right')
        all_patient_dataframe.loc[0, 'Abs_Energy'] = abs_en.calculate_abs_energy(patient_data[['Time_sec', 'TF_R']], 'TF_R')
        all_patient_dataframe = extract_eda_features(patient_data[['Time_sec', 'TF_R']], 'TF_R', all_patient_dataframe)
    return all_patient_dataframe


def add_foot_coeffs(all_patient_dataframe, fft, patient_data, feet_type):
    if feet_type == 'left':
        foot_coeff = fft.calculate_fft_coeff(patient_data[['Time_sec', 'TF_L']], 'TF_L')
    elif feet_type == 'right':
        foot_coeff = fft.calculate_fft_coeff(patient_data[['Time_sec', 'TF_R']], 'TF_R')
    else:
        raise ValueError("add_foot_coeffs() : Wrong value supplied")

    all_patient_dataframe.loc[0, 'coeff_1'] = foot_coeff['coeff_1__attr_"real"']
    all_patient_dataframe.loc[0, 'coeff_2'] = foot_coeff['coeff_2__attr_"real"']
    all_patient_dataframe.loc[0, 'coeff_3'] = foot_coeff['coeff_3__attr_"real"']
    all_patient_dataframe.loc[0, 'coeff_4'] = foot_coeff['coeff_4__attr_"real"']
    return all_patient_dataframe


def extract_eda_features(patient_data, col_name, all_patient_dataframe):
    all_patient_dataframe.loc[0, 'Median'] = tsfresh.feature_extraction.feature_calculators.median(patient_data[col_name])
    all_patient_dataframe.loc[0, 'Max'] = tsfresh.feature_extraction.feature_calculators.maximum(patient_data[col_name])
    all_patient_dataframe.loc[0, 'Min'] = tsfresh.feature_extraction.feature_calculators.minimum(patient_data[col_name])
    all_patient_dataframe.loc[0, 'Skewness'] = tsfresh.feature_extraction.feature_calculators.skewness(patient_data[col_name])
    all_patient_dataframe.loc[0, 'Std'] = tsfresh.feature_extraction.feature_calculators.standard_deviation(patient_data[col_name])
    all_patient_dataframe.loc[0, 'Variance'] = tsfresh.feature_extraction.feature_calculators.variance(patient_data[col_name])
    return all_patient_dataframe


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
