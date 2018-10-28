
"""
This script contains methods for loading data from files and writing to files
"""

# import packages or libraries
import pandas as pd

# import methods from other scripts


# Constant Declarations


# ----------------------------------------------- INPUT METHODS ------------------------------------------------------ #

class data_loader:

    def __init__(self):
        self.input_folder_path = "../input/"


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
