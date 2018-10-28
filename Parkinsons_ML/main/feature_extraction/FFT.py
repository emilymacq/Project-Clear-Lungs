
# import libraries
import tsfresh
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")



class fft_extractor:

    def calculate_fft_coeff(self, patient_data, col_name):
        fft_coeff_zipped = tsfresh.feature_extraction.feature_calculators.fft_coefficient(patient_data[col_name],
                                                                             [{"coeff": 1, "attr": 'real'},
                                                                              {"coeff": 2, "attr": 'real'},
                                                                              {"coeff": 3, "attr": 'real'},
                                                                              {"coeff": 4, "attr": 'real'}])

        fft_coeff_unzipped = dict(fft_coeff_zipped)
        return fft_coeff_unzipped

    def calculate_fft_agg(self, patient_data, col_name):
        fft_agg_zipped = tsfresh.feature_extraction.feature_calculators.fft_aggregated(patient_data[col_name],
                                                                                [{'aggtype': "centroid"},
                                                                                 {'aggtype': "variance"},
                                                                                 {'aggtype': "skew"},
                                                                                 {'aggtype': "kurtosis"}])

        fft_agg_unzipped = dict(fft_agg_zipped)
        return fft_agg_unzipped
