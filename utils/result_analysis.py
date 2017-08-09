import pandas as pd
import numpy as np
from collections import OrderedDict

def compute_beta_statistics(trace, beta, taxa, covariates, percentiles=[5, 95]):
    def convert_to_df(values):
        return pd.DataFrame(data=values, index=covariates, columns=taxa).T

    sheets = OrderedDict()
    sheets['beta_mean'] = convert_to_df(beta.mean(axis=0))
    sheets['beta_sd'] = convert_to_df(beta.std(axis=0))
    percentile_05, percentile_95 = np.percentile(beta, percentiles, axis=0)
    sheets['beta_05-percentile'] = convert_to_df(percentile_05)
    sheets['beta_95-percentile'] = convert_to_df(percentile_95)
    sheets['beta_select_confidence'] = convert_to_df(~((percentile_05 < 0) & (0 < percentile_95)))
    sheets['beta_mean-selected'] = sheets['beta_mean'].copy()
    sheets['beta_mean-selected'][~sheets['beta_select_confidence']] = 0
    # inclusion probability
    inclusion_probability = convert_to_df(compute_pseudo_inclusion_probability(trace))
    sheets['beta_pip'] = inclusion_probability

    # bfdr correction
    sheets['beta_select_bfdr'] = sheets['beta_mean'].copy()
    selected, threshold = bfdr(inclusion_probability.values, 0.1)
    sheets['beta_select_bfdr'][~selected] = 0
    return sheets


def generate_excel_summary(trace, taxa, covariates, outputfile):
    # beta trace has shape (#iterations, #covariates, #OTUs)
    beta = trace['beta']
    sheets = compute_beta_statistics(trace, beta, taxa, covariates)
    with pd.ExcelWriter(outputfile, engine='xlsxwriter') as excelfile:
        for sheet, data in sheets.items():
            data.to_excel(excelfile, sheet_name=sheet)


def compute_shrinkage_from_trace(trace):
    return (1/(1+ trace['lambda']**2 * (trace['tau']**2)[:, np.newaxis, np.newaxis])).mean(axis=0)


def compute_pseudo_inclusion_probability(trace):
    return 1 - compute_shrinkage_from_trace(trace)


def bfdr(inclusion_probabilities, threshold):
    assert np.all(inclusion_probabilities < 1) and np.all(inclusion_probabilities > 0)
    assert 0 < threshold < 1

    onek = ((1 - inclusion_probabilities) < threshold).astype(float)

    # possible to select none
    if np.sum(onek) == 0:
        selected = np.full(len(onek), False)
        thecut = 0
    else:
        thecut = np.sum((1 - inclusion_probabilities) * onek) / np.sum(onek)
        selected = (1 - inclusion_probabilities) < thecut

    return selected, 1 - thecut