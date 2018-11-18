
"""
This script contains methods for loading data from files and writing to files
"""

# import packages or libraries
import re
import glob
import ntpath
import pandas as pd

# import methods from other scripts


# Constant Declarations


# ----------------------------------------------- INPUT METHODS ------------------------------------------------------ #

class data_loader:

    def __init__(self):
        self.input_folder_path = "../resources/input/"

    def get_patient_file_paths(self):
        patient_file_paths = glob.glob(self.input_folder_path + 'patient_data/*')
        return patient_file_paths

    def split_file_path(self, file_path):
        """
        Splits the file path into two
        :param file_path: Path to the file
        :return: String with filename and extension
        """
        head, tail = ntpath.split(file_path)
        return tail or ntpath.basename(head)

    def extract_file_name(self, file_path):
        file_name = self.split_file_path(file_path)
        new_file_name = re.split('\.', file_name)  # list of 2 string elements: File name and extension
        return new_file_name[0]

    def load_demographics_data(self):
        """
        Loads Parkinsons demographics data
        :return: Pandas Dataframe
        """
        print("Loading demographic data...")
        demographics_file_name = "demographics.txt"
        demographics_file_path = self.input_folder_path + demographics_file_name
        demographics_data = self.read_txt_file(demographics_file_path, delim_whitespace=True, na_values='NaN')
        print("Data successfully loaded.")
        return demographics_data

    def read_txt_file(self, file_path, delim_whitespace, na_values):
        """
        Loads data from txt file
        :param file_path: path to input file from which data needs to be loaded
        :return: Pandas dataframe
        """
        txt_loaded_data = pd.read_csv(file_path, delim_whitespace=delim_whitespace, na_values=na_values)
        return txt_loaded_data

    def read_csv_file(self, file_path, delim_whitespace, na_values):
        """
        Loads data from csv file
        :param file_path: path to input file from which data needs to be loaded
        :return: Pandas dataframe
        """
        csv_loaded_data = pd.read_csv(file_path, delim_whitespace=delim_whitespace, na_values=na_values)
        return csv_loaded_data

    def read_patient_data(self, patient_file_path):
        col_names = ['Time_sec', 'Sens_L1', 'Sens_L2', 'Sens_L3', 'Sens_L4', 'Sens_L5', 'Sens_L6', 'Sens_L7', 'Sens_L8',
                     'Sens_R1', 'Sens_R2', 'Sens_R3', 'Sens_R4', 'Sens_R5', 'Sens_R6', 'Sens_R7', 'Sens_R8', 'TF_L',
                     'TF_R']
        patient_data = pd.read_csv(patient_file_path, header=None, sep='\t', names=col_names)
        return patient_data




