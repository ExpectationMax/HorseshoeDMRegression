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


def extract_patients_if_present(metadata, patientcol):
    if patientcol in metadata.columns:
        patients = metadata.patient
        metadata = metadata.drop(patientcol, axis=1)
        return patients, metadata
    else:
        return None, metadata


class Dataset:
    def __init__(self, countdata, metadata, patientcol='patient'):
        patients, metadata = extract_patients_if_present(metadata, patientcol)
        # self.metadata = center_and_standardize_columns(self.metadata)
        self.S, self.O = countdata.shape
        if patients is None:
            patients = list(range(self.S))
        _, self.C = metadata.shape
        self.taxa = countdata.columns.tolist()
        self.covariates = metadata.columns.tolist()
        self.metadata = metadata
        self.countdata = countdata
        self.patients = patients
