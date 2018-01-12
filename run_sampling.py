import os
import matplotlib
matplotlib.use('Agg')
import logging
logging.basicConfig(level=logging.DEBUG)

from utils.data import center_and_standardize_columns, extract_taxa_and_covariates, extract_patients_if_present
from utils.cli import verify_input_files, summarize_inputs, check_sainity, get_cli_parser, get_parameter_directories, \
    setup_logging
from utils.sampling import run_hmc_sampling
from utils.result_analysis import generate_excel_summary


def generate_outputs(model, trace, taxa, covariates, patients, n_samples, output, traceplot, save_model, save_trace):
    import pickle

    if save_trace:
        with open(os.path.join(output, 'trace.pck'), 'wb') as f:
            pickle.dump(trace, f, pickle.HIGHEST_PROTOCOL)

    if save_model:
        with open(os.path.join(output, 'model.pck'), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    if traceplot:
        import matplotlib.pyplot as plt
        import pymc3 as pm
        pm.traceplot(trace, varnames=['alpha', 'beta', 'tau'])
        plt.savefig(os.path.join(output, 'traceplot.pdf'))
        plt.close('all')

    generate_excel_summary(trace, taxa, covariates, patients, n_samples, os.path.join(output, 'variable_statistics.xlsx'))


def run_inference(countdata, metadata, estimated_covariates, output, transpose_counts, sampling_options, output_options):
    setup_logging(output)

    if transpose_counts:
        countdata = countdata.T

    verify_input_files(countdata, metadata)
    summarize_inputs(countdata, metadata, estimated_covariates)
    check_sainity(countdata, metadata)
    countdata = countdata.T # wider used notation (#samples, #features)
    patients, metadata = extract_patients_if_present(metadata, 'patient')
    metadata = center_and_standardize_columns(metadata)
    model, trace = run_hmc_sampling(countdata, metadata, patients, estimated_covariates, **sampling_options)
    taxa, covariates = extract_taxa_and_covariates(countdata, metadata)
    generate_outputs(model, trace, taxa, covariates, patients, countdata.shape[0], output, **output_options)


if __name__ == '__main__':
    parser = get_cli_parser()
    args = parser.parse_args()
    required_options, sampling_options, output_options = get_parameter_directories(args)
    run_inference(**required_options, sampling_options=sampling_options, output_options=output_options)



