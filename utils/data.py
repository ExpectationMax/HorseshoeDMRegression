def get_input_specs(countdata, metadata):
    O, S = countdata.shape
    S, C = metadata.shape
    return O, C, S


def extract_taxa_and_covariates(countdata, metadata):
    taxa = countdata.columns.tolist()
    covariates = metadata.columns.tolist()
    return taxa, covariates


def center_and_standardize_columns(data):
    return (data - data.mean(axis=0))/data.std(axis=0)


def extract_patients_if_present(metadata):
    if 'patient' in metadata.columns:
        patients = metadata.patient
        metadata = metadata.drop('patient', axis=1)
        return patients, metadata
    else:
        return None, metadata