import os
import shutil
import subprocess
import numpy as np
import pandas as pd

from utils.dmbvs import compute_alpha_init, compute_beta_init


basepath = os.path.dirname(os.path.abspath(__file__))
def run_dmbvs(metadata, countdata, GG, thin, burn, output_location, intercept_variance=10, slab_variance=10,
              bb_alpha=0.02, bb_beta=1.98, proposal_alpha=0.5, proposal_beta=0.5,
              executable = os.path.join(basepath, "lib","dmbvs.x"), r_seed = None, cleanup=True):
    os.makedirs(output_location, exist_ok=True)
    float_parameters = {'decimal':'.', 'index':False, 'header':False, 'sep':' ', 'float_format':'%.15g'}
    metadata.to_csv(os.path.join(output_location, 'covariates.txt'), **float_parameters)
    countdata.astype(int).to_csv(os.path.join(output_location, 'count_matrix.txt'), **float_parameters)

    nsamples = countdata.shape[0]
    notus = countdata.shape[1]
    ncovariates = metadata.shape[1]

    # initialization
    alpha_init = compute_alpha_init(countdata)
    alpha_init.to_csv(os.path.join(output_location, 'init_alpha.txt'), **float_parameters)

    beta_init = compute_beta_init(countdata, metadata)
    pd.DataFrame(beta_init.values.flatten()).to_csv(os.path.join(output_location, 'init_beta.txt'), **float_parameters)

    # write parameters
    pd.DataFrame([proposal_alpha]*countdata.shape[1]).to_csv(os.path.join(output_location, 'proposal_alpha.txt'), **float_parameters)
    pd.DataFrame([proposal_beta]*(countdata.shape[1]*metadata.shape[1]))\
        .to_csv(os.path.join(output_location, 'proposal_beta.txt'), **float_parameters)

    if r_seed == None:
        r_seed = np.random.randint(1, 1e6)

    proc = subprocess.Popen([executable] + list(map(str,
                                          [GG, thin, burn, intercept_variance, slab_variance, bb_alpha, bb_beta, notus,
                                           nsamples, ncovariates, r_seed, output_location]
                                           )
                                       ),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()

    # read results
    try:
        alpha = pd.read_table(os.path.join(output_location, 'alpha.out'), skipinitialspace=True, index_col=None, header=None, sep='\s+')
        beta = pd.read_table(os.path.join(output_location, 'beta.out'), skipinitialspace=True, index_col=None, header=None, sep='\s+')
    except Exception as e:
        print('Error reading dmbvs results!')
        print(e)
        print('dmbvs stout:')
        print(stdout)
        print('dmbvs stderr:')
        print(stderr)
        return None

    alpha.columns = countdata.columns
    alpha_mean = alpha.mean(axis=0)

    beta_reshaped = beta.values.reshape((-1, notus, ncovariates))
    beta_mean = pd.DataFrame(beta_reshaped.mean(axis=0), index=countdata.columns, columns=metadata.columns)
    mppip = pd.DataFrame((beta_reshaped != 0).mean(axis=0), index=countdata.columns, columns=metadata.columns)

    if cleanup:
        shutil.rmtree(output_location)

    return {'alpha': alpha_mean, 'beta':beta_mean, 'MPPI': mppip, 'alpha_trace':alpha, 'beta_trace': beta_reshaped, 'rseed': r_seed,
            'stdout': stdout, 'stderr':stderr}


def run_dmbvs_on_dataset_and_store(dataset, outputpath, GG=100000, burn=50000, thin=100):
    run_out = os.path.join(outputpath, dataset)
    dmbvs_tmp = os.path.join(run_out, 'dmbvs_tmp')

    if os.path.exists(os.path.join(run_out, 'results.pck')):
        return

    import pickle
    from data import get_simulated_data

    data = get_simulated_data(dataset, as_dataframe=True)
    os.makedirs(dmbvs_tmp, exist_ok=True)
    results = run_dmbvs(data['covariates'], data['counts'], GG, thin, burn, dmbvs_tmp, cleanup=False)
    if results is not None:
        with open(os.path.join(run_out, 'results.pck'), 'wb') as f:
            pickle.dump(results, f)
        del results


if __name__ == '__main__':
    import argparse
    from data import get_available_datasets
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser(description='Python wrapper for the dmbvs R package (Wadsworth, et al. An Integrative Bayesian Dirichlet-Multinomial Regression Model for the Analysis of Taxonomic Abundances in Microbiome data. (2016))')
    parser.add_argument('datasets', nargs='+', choices=get_available_datasets(), help='Datasets on which dmbvs should be run')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output folder for traces generated using dmbvs.')
    parser.add_argument('--njobs', default=1, type=int, help='Number of dataset to run in parallel.')
    args = parser.parse_args()

    njobs = len(args.datasets)
    Parallel(n_jobs=njobs)(delayed(run_dmbvs_on_dataset_and_store)(dataset, args.output) for dataset in args.datasets)
