import argparse
from utils.cli import tsv_file, pickle_file, nonexistant_file
from utils.data import extract_taxa_and_covariates
from utils.result_analysis import generate_excel_summary


def generate_analysis_output(countdata, metadata, trace, outputfile):
    countdata = countdata.T
    taxa, covariates = extract_taxa_and_covariates(countdata, metadata)
    generate_excel_summary(trace, taxa, covariates, outputfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('countdata', type=tsv_file)
    parser.add_argument('metadata', type=tsv_file)
    parser.add_argument('trace', type=pickle_file)
    parser.add_argument('outputfile', type=nonexistant_file)
    args = parser.parse_args()

    generate_analysis_output(args.countdata, args.metadata, args.trace, args.outputfile)
