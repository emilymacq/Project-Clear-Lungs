
# import libraries
import tsfresh
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")



class AbsoluteEnergyCalculator:

    def calculate_abs_energy(self, patient_data, col_name):
        abs_energy = tsfresh.feature_extraction.feature_calculators.abs_energy(patient_data[col_name])
        abs_energy_scaled = abs_energy / pow(10, 6)
        return abs_energy_scaled
