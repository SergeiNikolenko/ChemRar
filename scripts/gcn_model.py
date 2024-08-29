# %%
import torch
import pandas as pd
import torch

import warnings
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parents[0]))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR

from chemrar_test.prepare import MoleculeDataset
if torch.cuda.is_available():
    print("CUDA available:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("CUDA is not available.")

torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")


# %%
print("Starting to load dataset...")

dataset = torch.load("../data/processed/data_graph.pt")

print("Dataset loaded successfully. Length of dataset:", len(dataset))

molecule_dataset = dataset
print("First element of the dataset:", molecule_dataset[0])



# %%
from tqdm import tqdm

print("Loading split data...")
split_path = "/home/nikolenko/work/Project/ChemRar/data/processed/random_split.csv"
split_df = pd.read_csv(split_path)
print("Split data loaded successfully.")

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

print("Splitting dataset into train, validation, and test sets...")

train_smiles_set = set(split_df['Train_SMILES'])
val_smiles_set = set(split_df['Val_SMILES'])
test_smiles_set = set(split_df['Test_SMILES'])

train_data = []
val_data = []
test_data = []

for data in tqdm(molecule_dataset, desc="Splitting dataset"):
    smiles = data.smiles

    if smiles in train_smiles_set:
        train_data.append(data)
    elif smiles in val_smiles_set:
        val_data.append(data)
    elif smiles in test_smiles_set:
        test_data.append(data)

print(f"Split completed. Train set: {len(train_data)}, Val set: {len(val_data)}, Test set: {len(test_data)}")


# %%
class AtomEdgeInteraction(nn.Module):
    def __init__(self, in_features, edge_features, out_features, edge_importance=1.0, dropout_rate=0.1, use_batch_norm=True):
        super(AtomEdgeInteraction, self).__init__()
        self.edge_importance = edge_importance
        self.interaction = nn.Linear(in_features + edge_features, out_features)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        row = row.long()
        col = col.long()
        edge_attr = edge_attr.float()
        x = x.float()
        edge_features = edge_attr * self.edge_importance
        atom_features = x[row]
        combined_features = torch.cat([atom_features, edge_features], dim=-1)
        updated_features = self.interaction(combined_features)
        updated_features = self.activation(updated_features)
        updated_features = self.batch_norm(updated_features)
        updated_features = self.dropout(updated_features)
        residual_features = self.residual(x)
        x = scatter_mean(updated_features, col, dim=0, dim_size=x.size(0))
        return x + residual_features

class SimplifiedMoleculeModel(pl.LightningModule):
    def __init__(self, atom_in_features, edge_in_features, hidden_features, dropout_rates, out_features, learning_rate, weight_decay, batch_size, linear_layer_sizes, step_size, gamma):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.test_outputs = []

        self.atom_edge_interaction1 = AtomEdgeInteraction(
            in_features=atom_in_features,
            edge_features=edge_in_features,
            out_features=hidden_features,
            dropout_rate=dropout_rates
        )

        self.gcn_conv1 = GCNConv(
            in_channels=hidden_features, 
            out_channels=hidden_features
        )

        self.postprocess = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ELU(),
            nn.Dropout(dropout_rates),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_edge_interaction1(x, edge_index, edge_attr)
        x = self.gcn_conv1(x, edge_index.long())
        x = nn.ELU()(x)
        x = self.postprocess(x).squeeze(-1)
        return x
    
    def training_step(self, batch, batch_idx):
        batch.batch = batch.batch.long()
        x = self(batch.x, batch.edge_index, batch.edge_attr)
        y_hat = scatter_mean(x, batch.batch, dim=0)
        loss = nn.BCEWithLogitsLoss()(y_hat, batch.y.float())
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch.batch = batch.batch.long()
        x = self(batch.x, batch.edge_index, batch.edge_attr)
        y_hat = scatter_mean(x, batch.batch, dim=0)
        val_loss = nn.BCEWithLogitsLoss()(y_hat, batch.y.float())
        self.log('val_loss', val_loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        batch.batch = batch.batch.long()
        x = self(batch.x, batch.edge_index, batch.edge_attr)
        y_hat = scatter_mean(x, batch.batch, dim=0)
        y_pred = torch.sigmoid(y_hat).cpu().numpy()
        y_true = batch.y.float().cpu().numpy()
        self.test_outputs.append({'y_pred': y_pred, 'y_true': y_true})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def on_test_epoch_end(self):
        y_pred = np.concatenate([o['y_pred'] for o in self.test_outputs])
        y_true = np.concatenate([o['y_true'] for o in self.test_outputs])

        accuracy = accuracy_score(y_true, y_pred.round())
        precision = precision_score(y_true, y_pred.round(), zero_division=0)
        recall = recall_score(y_true, y_pred.round(), zero_division=0)
        f1 = f1_score(y_true, y_pred.round(), zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
        else:
            roc_auc = float('nan')
            pr_auc = float('nan')
        
        self.log_dict({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'test_pr_auc': pr_auc
        })
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}')


# %%
logger = TensorBoardLogger("../reports/gcn_log", name="molecule_model")

from pytorch_lightning.callbacks import EarlyStopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=True
)
trainer = pl.Trainer(
    max_epochs=100,
    log_every_n_steps=10,
    logger=logger,
    callbacks=[early_stop_callback],
    enable_progress_bar=False

)

atom_in_features = dataset.num_features
edge_in_features = dataset.num_edge_features
hidden_features = 128
dropout_rates = 0.2
out_features = 1
learning_rate = 0.001
weight_decay = 1e-5
batch_size = 1024
linear_layer_sizes = [128, 128] 
step_size = 20
gamma = 0.1

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = SimplifiedMoleculeModel(
    atom_in_features=atom_in_features,
    edge_in_features=edge_in_features,
    hidden_features=hidden_features,
    dropout_rates=dropout_rates,
    out_features=out_features,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    batch_size=batch_size,
    linear_layer_sizes=linear_layer_sizes,
    step_size=step_size,
    gamma=gamma
)

# %%
trainer.fit(model, train_loader, val_loader)

# %%
trainer.test(model, test_loader)

# %%
torch.save(model.state_dict(), '../models/gcn_model.pth')


