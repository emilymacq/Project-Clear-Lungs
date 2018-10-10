"""
This script is to perform initial analysis on the demographics and patients data.
Objective:
    1. Look at demographics and type of data
    2. Understand how the patients are distributed over different studies
    3. Decide which files to load for machine learning algorithms
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# Print Settings
pd.set_option('display.expand_frame_repr', False)


def print_newline():
    print("")


def print_seperator():
    print("--------------------------")


def plot_sensor_positions():
    """
    Plots the position of sensors that are put on patient's feet.
    The positions are given in - input/format.txt
    :return: None
    """

    left_foot_x = ['-500', '-700', '-300', '-700', '-300', '-700', '-300', '-500']
    left_foot_y = ['-800', '-400', '-400', '0', '0', '400', '400', '800']
    right_foot_x = ['500', '700', '300', '700', '300', '700', '300', '500']
    right_foot_y = ['-800', '-400', '-400', '0', '0', '400', '400', '800']

    sens_pos = pd.DataFrame({'LX': left_foot_x, 'LY': left_foot_y, 'RX': right_foot_x, 'RY': right_foot_y})

    cols = sens_pos.columns
    sens_pos[cols] = sens_pos[cols].apply(pd.to_numeric, errors='coerce', axis=1)

    ax = sns.scatterplot(x='LX', y='LY', data=sens_pos)
    sns.scatterplot(ax=ax, x='RX', y='RY', data=sens_pos)

    ax.set_title('Position of Sensors')
    ax.set_ylabel('Y-Coordinate')
    ax.set_xlabel('X-Coordinate')

    plt.show()


def print_demographics_info(demographics_data):
    # print demographics info
    print_newline()
    print_seperator()
    print("Demographics info:\n")
    demographics_data.info()
    print_seperator()


def drop_demographics_columns(demographics_data):
    col_to_drop = ['Speed_02', 'Speed_03', 'Speed_04', 'Speed_05', 'Speed_06', 'Speed_07', 'Speed_10']
    demographics_data = demographics_data.drop(col_to_drop, axis=1)
    return demographics_data


def get_group_study_types(demographics_data):
    # Find the number of groups
    group_types = demographics_data['Group'].unique()
    print('\nDifferent groups are: ', group_types)

    # Find the number of Studies
    study_types = demographics_data['Study'].unique()
    print('\nDifferent studies are: ', study_types)

    return group_types, study_types


def plot_study_group_data(demographics_data):
    ax_1 = sns.countplot(x="Group", data=demographics_data)
    ax_1.set_title("# of participants in each group")

    ax_2 = sns.catplot(x="Study", col="Group", kind="count", data=demographics_data)
    ax_2.set_titles("# of participants per study for Group: {col_name}")
    plt.show()


def correct_height_values(demographics_data):
    demographics_data['Height'] = demographics_data['Height'].apply(lambda x: x / 100 if x > 100 else x)
    return demographics_data


def print_demographics_EDA(demographics_data):
    # print demographics EDA
    print_newline()
    print_seperator()
    print("Demographics EDA:\n")
    print(demographics_data.describe())
    print_seperator()


def print_participant_count(demographics_data):
    print_newline()
    print_seperator()
    print("Participant count per study per group:\n")
    print(pd.DataFrame(demographics_data.groupby(['Group', 'Study'])['Subjnum'].count()))
    print_seperator()


def plot_parkinson_measures(demographic_data):
    ax_1 = sns.relplot(x='HoehnYahr', y='Subjnum', col='Group', row='Study', data=demographic_data)
    ax_2 = sns.relplot(x='TUAG', y='Subjnum', col='Group', row='Study', data=demographic_data)
    plt.show()


# ----------------------------------------------------- GROUP 1 ------------------------------------------------------ #


def group_1_analysis(group_1_data):
    print_newline()
    print("#####################################")
    print("Group 1 Analysis:")
    print("#####################################")

    print_newline()
    print_seperator()
    print("Group 1 info:\n")
    group_1_data.info()
    print_seperator()

    plot_age_distribution(group_1_data, "Age distribution in Group 1")
    plot_participants_height(group_1_data)

    col_to_consider = ['Age', 'Height', 'Weight', 'HoehnYahr', 'UPDRS', 'UPDRSM', 'TUAG', 'Speed_01']
    no_nan_data = group_1_data[col_to_consider].dropna()
    grp_corr_matrix = no_nan_data.corr()
    plot_corr_heatmap(grp_corr_matrix, "Relation b/w attributes in Group 1")


# ----------------------------------------------------- GROUP 2 ------------------------------------------------------ #

def group_2_analysis(group_2_data):
    print_newline()
    print("#####################################")
    print("Group 2 Analysis:")
    print("#####################################")

    print_newline()
    print_seperator()
    print("Group 2 info:\n")
    group_2_data.info()
    print_seperator()

    plot_age_distribution(group_2_data, "Age distribution in Group 2")
    plot_participants_height(group_2_data)

    col_to_consider = ['Age', 'Height', 'Weight', 'UPDRS', 'UPDRSM', 'TUAG', 'Speed_01']
    no_nan_data = group_2_data[col_to_consider].dropna()
    grp_corr_matrix = no_nan_data.corr()
    plot_corr_heatmap(grp_corr_matrix, "Relation b/w attributes in Group 2")


# -------------------------------------------------- PLOTTING METHODS ------------------------------------------------ #


def plot_age_distribution(group_data, plot_title):
    # To get the distribution of participant age in group 1
    ax = sns.countplot(x="Age", data=group_data)
    ax.set_title(plot_title)
    plt.show()


def plot_participants_height(group_data):
    # To get the distribution of participant height in group 1
    ax = sns.relplot(x="Subjnum", y="Height", col="Study", data=group_data)
    plt.show()


def plot_corr_heatmap(group_corr_matrix, plot_title):
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(group_corr_matrix, annot=True, cbar=True, cmap=cmap, fmt='.2f')
    ax.set_title(plot_title)
    plt.show()


# ------------------------------------------------ ANALYSIS CONTROLLER------------------------------------------------ #

def analysis_controller(demographics_data):
    plot_sensor_positions()
    print_demographics_info(demographics_data)

    demographics_data = drop_demographics_columns(demographics_data)

    group_types, study_types = get_group_study_types(demographics_data)
    plot_study_group_data(demographics_data)

    demographics_data = correct_height_values(demographics_data)

    print_demographics_EDA(demographics_data)
    print_participant_count(demographics_data)
    plot_parkinson_measures(demographics_data)


