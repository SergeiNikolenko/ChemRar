# %%
import pandas as pd
from sklearn.model_selection import train_test_split



dataset = pd.read_csv("/home/nikolenko/work/Project/ChemRar/data/raw/data.csv")


smiles = dataset['SMILES'].tolist()


# %%

train_smiles, temp_smiles = train_test_split(smiles, train_size=0.6, random_state=42)


val_smiles, test_smiles = train_test_split(temp_smiles, test_size=0.5, random_state=42)

print(f"Random Split - Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")


# %%
# Random Split
random_split_df = pd.DataFrame({
    'Train_SMILES': pd.Series(train_smiles),
    'Val_SMILES': pd.Series(val_smiles),
    'Test_SMILES': pd.Series(test_smiles)
})

random_split_df.to_csv("/home/nikolenko/work/Project/ChemRar/data/processed/random_split.csv", index=False)

