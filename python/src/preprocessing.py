import pandas as pd
from pathlib import Path
import numpy as np


def load_data(dataset_path, target=True, images=False):
    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    radiomics_file = dataset_path / "features" / "radiomics.csv"
    columns = ["PatientID"] + list(pd.read_csv(radiomics_file, skiprows=1).columns[1:])
    radiomics = pd.read_csv(radiomics_file, skiprows=3, names=columns, index_col='PatientID')

    clinical_file = dataset_path / "features" / "clinical_data.csv"
    clinical = pd.read_csv(clinical_file, index_col='PatientID')
    clinical.loc[:, 'Histology'] = clinical.Histology.str.lower()
    clinical.loc[clinical.Histology == 'nsclc nos (not otherwise specified)', 'Histology'] = 'nos'

    if target:
        target_file = dataset_path / "y_train.csv"
        target = pd.read_csv(target_file, index_col='PatientID')

        return radiomics, clinical, target
    else:
        return radiomics, clinical


def age_fillna(row, svr_age, pipeline):
    scaler = pipeline.named_transformers_.rescale_age_column
    if np.isnan(row.age) and not row[:-1].isna().any():
        X = pipeline.transform(pd.DataFrame(row).T)
        row.age = scaler.inverse_transform(svr_age.predict(X[:, :-1]).reshape(-1, 1))[0][0]
        return row
    else:
        return row


def hist_fillna(row, svc_hist, pipeline):
    encoder = pipeline.named_transformers_.encode_histology_column
    if row.Histology is np.NaN and not row[1:].isna().any():
        # add a value to avoid nan in pipeline
        row.Histology = 'adenocarcinoma'
        X = pipeline.transform(pd.DataFrame(row).T)
        row.Histology = encoder.inverse_transform(svc_hist.predict(X[:, 1:]).reshape(-1, 1))[0][0]
        return row
    else:
        return row
