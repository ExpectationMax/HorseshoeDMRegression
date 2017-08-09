import pandas as pd
import numpy as np
from collections import OrderedDict

def compute_beta_statistics(beta, taxa, covariates, percentiles=[5, 95]):
    def convert_to_df(values):
        return pd.DataFrame(data=values, index=covariates, columns=taxa).T

    sheets = OrderedDict()
    sheets['beta_mean'] = convert_to_df(beta.mean(axis=0))
    sheets['beta_sd'] = convert_to_df(beta.std(axis=0))
    percentile_05, percentile_95 = np.percentile(beta, percentiles, axis=0)
    sheets['beta_05-percentile'] = convert_to_df(percentile_05)
    sheets['beta_95-percentile'] = convert_to_df(percentile_95)
    sheets['beta_select'] = convert_to_df(~((percentile_05 < 0) & (0 < percentile_95)))
    sheets['beta_mean-selected'] = sheets['beta_mean'].copy()
    sheets['beta_mean-selected'][~sheets['beta_select']] = 0

    return sheets


def generate_excel_summary(trace, taxa, covariates, outputfile):
    # beta trace has shape (#iterations, #covariates, #OTUs)
    beta = trace['beta']
    sheets = compute_beta_statistics(beta, taxa, covariates)
    with pd.ExcelWriter(outputfile, engine='xlsxwriter') as excelfile:
        for sheet, data in sheets.items():
            data.to_excel(excelfile, sheet_name=sheet)


def compute_shrinkage_from_trace(trace):
    pass