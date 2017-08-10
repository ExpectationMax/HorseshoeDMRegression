import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from .data import get_input_specs

def tsv_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))

    try:
        data = pd.read_table(x, index_col=0, header=0)
    except Exception as e:
        raise argparse.ArgumentTypeError("{} does not seem to be a valid tsv file:\n{}".format(x, e))

    return data


def pickle_file(x):
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))

    try:
        import pickle
        with open(x, 'rb') as file:
            data = pickle.load(file)
    except Exception as e:
        raise argparse.ArgumentTypeError("{} does not seem to be a valid pickled file:\n{}".format(x, e))

    return data


def nonexistant_file(x):
    if os.path.exists(x):
        raise argparse.ArgumentTypeError("{} exists and would be overwritten.".format(x))
    return x


def summarize_inputs(countsdata, metadata, p0):
    summary = logging.getLogger('Input summary')
    O, C, S = get_input_specs(countsdata, metadata)
    summary.info('OTUs: %i, Covariates: %i, Samples: %i', O, C, S)
    summary.info('Looking for associations with following covariates: %s', metadata.columns.tolist())
    summary.info('{} covariates are estimated to be present in the data (Sparcity estimate)'.format(p0))


def check_sainity(countdata, metadata):
    O, C, S = get_input_specs(countdata, metadata)
    checker = logging.getLogger('Sainity check')
    if O/S > 1.4:
        checker.warning('The number of OTUs is significantly higher than the number of samples! Inference might be unstable.')


def verify_input_files(countdata, metadata):
    if not np.all(countdata.columns == metadata.index):
        raise(ValueError('Columns of countdata and metadata differ! Please check the input files.'))


def get_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('countdata', type=tsv_file)
    parser.add_argument('metadata', type=tsv_file)
    parser.add_argument('--estimated_covariates', type=int, required=True)
    parser.add_argument('-o', '--output', required=True, type=nonexistant_file)

    sampling_group = parser.add_argument_group('Sampling options')
    sampling_group.add_argument('--n_chains', type=int, default=4)
    sampling_group.add_argument('--n_tune', type=int, default=2000)
    sampling_group.add_argument('--n_draws', type=int, default=2000)
    sampling_group.add_argument('--seed', type=int, default=-1)

    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--traceplot', action='store_true', default=False)
    output_group.add_argument('--save_model', action='store_true', default=False)
    output_group.add_argument('--save_trace', action='store_true', default=False)

    return parser


def get_parameter_directories(args):
    required_options = {'countdata': args.countdata, 'metadata': args.metadata,
                        'estimated_covariates': args.estimated_covariates, 'output':args.output}
    sampling_options = {'n_chains': args.n_chains, 'n_tune':args.n_tune, 'n_draws':args.n_draws, 'seed': args.seed}
    output_options = {'traceplot': args.traceplot, 'save_model': args.save_model, 'save_trace':args.save_trace}
    return required_options, sampling_options, output_options


def log_except_hook(*exc_info):
    import traceback
    text = "".join(traceback.format_exception(*exc_info))
    logging.error("Unhandled exception: %s", text)


def setup_logging(outputdir):
    os.makedirs(outputdir)
    logging.root.addHandler(logging.FileHandler(os.path.join(outputdir, 'sampling.log')))
    sys.excepthook = log_except_hook