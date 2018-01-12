import os
import pickle
import pandas as pd
from glob import glob


def get_sucessful_runs(dataset, resultspath='results'):
    datafiles = [os.path.basename(filepath)[:-len('_sampling.pck')] for filepath in glob(os.path.join(resultspath, dataset, '*_sampling.pck'))]
    return datafiles


def get_model_data(dataset, modelname, resultspath='results'):
    inputfile = os.path.join(resultspath, dataset, '{}_sampling.pck'.format(modelname))
    with open(inputfile, 'rb') as f:
        trace = pickle.load(f)

    return trace


def get_sample_size_from_dataset(dataset):
    candidates = [frag for frag in dataset.split('_') if frag[-1] == 'S']
    assert len(candidates) == 1
    samplestr = candidates[0]
    return int(samplestr[:-1])


model_lookup = {}
def split_modelname_into_parameters(data):
    model_parameters = ['Type', 'Nu', 'Parametrization', 'Hyperprior', 'Model p0']
    def split_model_name(modelname):
        if modelname in model_lookup.keys():
            return model_lookup[modelname]
        fragments = modelname.split('_')
        if len(fragments) == 2:
            ret = pd.Series(index=model_parameters, data=[fragments[1], '', '', '', ''])
        else:
            ret = pd.Series(index=model_parameters, data=[fragments[0], int(fragments[1][2:]), fragments[2], fragments[3], int(fragments[4][2:])])
        model_lookup[modelname] = ret
        return ret

    return model_parameters, data.apply(split_model_name)


dataset_lookup = {}
def split_datasetname_into_parameters(data):
    dataset_parameters = ['OTUs', 'Covariates', 'Data p0', 'Samples', 'Repetition']
    def split_dataset_name(dataset):
        if dataset in dataset_lookup.keys():
            return dataset_lookup[dataset]
        frags = dataset.split('_')
        ret = pd.Series(index=dataset_parameters,
                        data=[int(frags[0][:-1]), int(frags[1][:-1]), int(frags[2][:-2]), int(frags[3][:-1]), int(frags[4][:-1])])
        dataset_lookup[dataset] = ret
        return ret

    return dataset_parameters, data.apply(split_dataset_name)


