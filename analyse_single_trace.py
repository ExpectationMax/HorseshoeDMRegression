import argparse
from utils.cli import tsv_file, pickle_file, nonexistant_file
from utils.data import Dataset
from utils.result_analysis import generate_excel_summary


def generate_analysis_output(countdata, metadata, trace, outputfile, transpose):
    if transpose:
        countdata = countdata.T

    dataset = Dataset(countdata, metadata)
    generate_excel_summary(trace, dataset, outputfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('countdata', type=tsv_file, help='Read counts associated with individual microbes in tab separated format with layout Samples (rows) x Microbes (columns).')
    parser.add_argument('metadata', type=tsv_file, help='Covariates associated with samples in tab separated format with layout Samples (rows) x Covariates (columns).')
    parser.add_argument('trace', type=pickle_file, help='Trace file generated during sampling.')
    parser.add_argument('outputfile', type=nonexistant_file, help='File for excel trace summary.')
    parser.add_argument('--transpose-counts', action='store_true', default=False, help='Accept read counts in tab separated format with Microbes (rows) x Samples (columns) layout instead.')
    args = parser.parse_args()

    generate_analysis_output(args.countdata, args.metadata, args.trace, args.outputfile, args.transpose_counts)
