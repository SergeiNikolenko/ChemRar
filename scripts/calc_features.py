import os
import numpy as np
import pandas as pd
import datamol as dm
from pathlib import Path

from molfeat.calc import FPCalculator
from molfeat.trans import FPVecTransformer

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
dm.parallelized
from tqdm import tqdm

import pyarrow.parquet as pq
import pyarrow as pa

dm.disable_rdkit_log()

data = pd.read_csv('data/raw/data.csv')
print("Data loaded")


smiles_column = "SMILES"
def create_featurizers(length=2048):
    featurizers = {}

    parametric_types = ["ecfp"]
    for pt in parametric_types:
        for size in range(4, 5):
            key = f"{pt}:{size}"
            featurizers[key] = FPVecTransformer(key, n_jobs=50, length=length, dtype=np.float32, parallel_kwargs={'progress': True})

    return featurizers

featurizers = create_featurizers()

os.makedirs("../data/processed", exist_ok=True)

data_unique = data.drop_duplicates(subset=[smiles_column]).copy()
data_unique["mol"] = data_unique[smiles_column].parallel_apply(dm.to_mol)
mols = data_unique["mol"].tolist()
smiles = data_unique[smiles_column].tolist()
print("Количество уникальных структур после удаления дубликатов:", len(mols))

def batch_generator(data, batch_size):
    return (data[i:i+batch_size] for i in range(0, len(data), batch_size))

def save_features(featurizers, molecules, smiles, batch_size=10_000):
    total_molecules = len(molecules)
    with tqdm(total=total_molecules, desc="Overall Progress") as pbar:
        for key, featurizer in featurizers.items():
            schema = None
            writer = None
            for mol_batch, smiles_batch in zip(batch_generator(molecules, batch_size), batch_generator(smiles, batch_size)):
                features = featurizer(mol_batch)
                df = pd.DataFrame({
                    smiles_column: smiles_batch,
                    f"{key}_features": list(features)
                })
                table = pa.Table.from_pandas(df)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(f"data/processed/{key}_features.parquet", schema, compression='snappy')
                writer.write_table(table)
                pbar.update(len(mol_batch))
            if writer:
                writer.close()

save_features(featurizers, mols, smiles)