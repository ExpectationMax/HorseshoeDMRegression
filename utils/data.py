def get_input_specs(countdata, metadata):
    O, S = countdata.shape
    S, C = metadata.shape
    return O, C, S


def extract_taxa_and_covariates(countdata, metadata):
    taxa = countdata.index.tolist()
    covariates = metadata.columns.tolist()
    return taxa, covariates

def center_and_standardize_columns(data):
    return (data - data.mean(axis=0))/data.std(axis=0)