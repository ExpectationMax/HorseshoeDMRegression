import argparse
import pickle
import pandas as pd
from utils.dmbvs import read_results
from combine_hmc_results import split_datasetname_into_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--n_jobs', default=10, type=int)

    args = parser.parse_args()
    failed, results = read_results(args.results_folder, args.n_jobs)

    print('failed datasets:')
    print(failed)

    _, res_dataset_details = split_datasetname_into_parameters(results['Dataset'])
    res = pd.concat([res_dataset_details, results], axis=1)

    with open(args.output, 'wb') as f:
        pickle.dump(res, f)
