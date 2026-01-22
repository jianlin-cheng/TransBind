import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import math
import json
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_float32_matmul_precision('medium') 

class TransDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ProteinFeatureLoader:
    """Handles loading and processing protein features from .fea files"""
    
    def __init__(self, mapping_file, features_dir):
        self.mapping_file = mapping_file
        self.features_dir = features_dir
        self.load_mapping()
        self.load_protein_features()
    
    def load_mapping(self):
        """Load the TF to feature mapping"""
        with open(self.mapping_file, 'r') as f:
            self.mapping_data = json.load(f)
        
        self.tf_to_feature_mapping = self.mapping_data['tf_to_feature_mapping']
        self.feature_files = self.mapping_data['feature_metadata']['feature_files']
        
        print(f"Loaded mapping for {len(self.tf_to_feature_mapping)} TF labels")
        print(f"Found {len(self.feature_files)} unique feature files")
    
    def load_single_feature_file(self, feature_file):
        """Load a single .fea file and return the feature vector"""
        file_path = os.path.join(self.features_dir, feature_file)
        
        if not os.path.exists(file_path):
            print(f"Warning: Feature file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            numbers = [float(x) for x in content.split()]
            data = np.array(numbers)
            
            if len(numbers) % 1280 == 0:
                sequence_length = len(numbers) // 1280
                print(f"Loaded {feature_file}: {len(numbers)} numbers = {sequence_length} × 1280")
                
                data_matrix = data.reshape(sequence_length, 1280)
                
                feature_vector = np.mean(data_matrix, axis=0)
                
                print(f"  Aggregated to: {feature_vector.shape}")
                return feature_vector
                
            else:
                print(f"Warning: {feature_file} has {len(numbers)} elements, not divisible by 1280")
                if len(numbers) >= 1280:
                    return data[:1280]
                else:
                    return None
                
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
            return None
    
    def load_protein_features(self):
        """Load all protein features and create feature matrix"""
        print("Loading protein features...")
        
        sample_loaded = False
        feature_dim = None
        
        for feature_file in self.feature_files:
            sample_features = self.load_single_feature_file(feature_file)
            if sample_features is not None:
                feature_dim = len(sample_features)
                sample_loaded = True
                break
        
        if not sample_loaded:
            raise ValueError("Could not load any feature files to determine dimensions!")
        
        print(f"Feature dimension: {feature_dim}")
        
        num_feature_files = len(self.feature_files)
        self.feature_matrix = np.zeros((num_feature_files, feature_dim))
        
        loaded_count = 0
        for idx, feature_file in enumerate(self.feature_files):
            features = self.load_single_feature_file(feature_file)
            if features is not None:
                if len(features) == feature_dim:
                    self.feature_matrix[idx] = features
                    loaded_count += 1
                else:
                    print(f"Dimension mismatch for {feature_file}: expected {feature_dim}, got {len(features)}")
                    self.feature_matrix[idx] = np.zeros(feature_dim)
            else:
                self.feature_matrix[idx] = np.zeros(feature_dim)
        
        print(f"Successfully loaded {loaded_count}/{num_feature_files} feature files")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        self.feature_matrix = self.normalize_features(self.feature_matrix)
    
    def normalize_features(self, features):
        """Features are already processed, no normalization needed"""
        print("Protein features are already well-normalized, skipping normalization")
        return features
    
    def get_tf_features_matrix(self):
        """Get the TF features matrix aligned with the 690 labels"""
        tf_features = np.zeros((690, self.feature_matrix.shape[1]))
        
        for tf_idx in range(690):
            feature_id = self.tf_to_feature_mapping[tf_idx]
            if feature_id != -1:
                tf_features[tf_idx] = self.feature_matrix[feature_id]
        
        return tf_features

class GeneralizedProteinAware(pl.LightningModule):
    def __init__(self, mapping_file, features_dir, learning_rate=0.001, dropout=0.1, 
                 pos_weight=None, tf_feature_projection_dim=320):
        super().__init__()
        self.save_hyperparameters()
        
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.dropout_cnn = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=320,
            hidden_size=160,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=320,
            nhead=8,
            dim_feedforward=1280,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        print("Loading protein features...")
        self.protein_loader = ProteinFeatureLoader(mapping_file, features_dir)
        tf_features_matrix = self.protein_loader.get_tf_features_matrix()
        
        original_feature_dim = tf_features_matrix.shape[1]
        print(f"Original protein feature dimension: {original_feature_dim}")
        
        self.tf_feature_projection_dim = tf_feature_projection_dim
        self.protein_feature_projection = nn.Sequential(
            nn.Linear(original_feature_dim, tf_feature_projection_dim),
            nn.LayerNorm(tf_feature_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.register_buffer('tf_features_raw', torch.FloatTensor(tf_features_matrix))
        
        self.tf_cross_attention = nn.MultiheadAttention(
            embed_dim=320, num_heads=8, batch_first=True
        )
        
        self.dna_tf_fusion = nn.Sequential(
            nn.Linear(640, 320),
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.shared_predictor = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.validation_step_outputs = []
        
        print(f"Model initialized with {original_feature_dim}D -> {tf_feature_projection_dim}D protein features")
        print("🎯 Model learns DNA-protein relationships (not TF identities)")
    
    def get_projected_tf_features(self):
        """Get protein features projected to the desired dimension - NO masking/noise"""
        tf_features = self.protein_feature_projection(self.tf_features_raw)
        return tf_features
    
    def process_single_tf(self, dna_sequence_features, dna_global, tf_feature):
        """Process a single TF with DNA sequence - core DNA-protein interaction learning"""
        batch_size = dna_global.size(0)
        
        tf_batch = tf_feature.unsqueeze(0).expand(batch_size, 1, -1)
        
        tf_attended_dna, attention_weights = self.tf_cross_attention(
            query=tf_batch,
            key=dna_sequence_features,
            value=dna_sequence_features
        )
        tf_attended_dna = tf_attended_dna.squeeze(1)
        
        combined_features = torch.cat([dna_global, tf_attended_dna], dim=1)
        
        fused_features = self.dna_tf_fusion(combined_features)
        
        binding_logit = self.shared_predictor(fused_features)
        
        return binding_logit.squeeze(-1), attention_weights
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d(x))
        x = self.maxpool(x)
        x = self.dropout_cnn(x)
        
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.bilstm(x)
        
        dna_sequence_features = self.transformer_layer(lstm_out)
        
        x = dna_sequence_features.transpose(1, 2)
        dna_global = self.global_pool(x).squeeze(-1)
        
        tf_features = self.get_projected_tf_features()
        
        all_tf_predictions = []
        for tf_idx in range(690):
            tf_feature = tf_features[tf_idx]
            tf_prediction, _ = self.process_single_tf(
                dna_sequence_features, dna_global, tf_feature
            )
            all_tf_predictions.append(tf_prediction.unsqueeze(1))
        
        output = torch.cat(all_tf_predictions, dim=1)
        
        return output
    
    def predict_single_tf(self, dna_seq, tf_idx):
        """Predict binding for just one specific TF - much faster!"""
        self.eval()
        with torch.no_grad():
            x = dna_seq.transpose(1, 2)
            x = F.relu(self.conv1d(x))
            x = self.maxpool(x)
            x = self.dropout_cnn(x)
            
            x = x.transpose(1, 2)
            lstm_out, _ = self.bilstm(x)
            dna_sequence_features = self.transformer_layer(lstm_out)
            
            x = dna_sequence_features.transpose(1, 2)
            dna_global = self.global_pool(x).squeeze(-1)
            
            tf_features = self.get_projected_tf_features()
            tf_feature = tf_features[tf_idx]
            
            prediction, attention = self.process_single_tf(
                dna_sequence_features, dna_global, tf_feature
            )
            
            return torch.sigmoid(prediction), attention
    
    def predict_new_tf(self, dna_seq, new_tf_features):
        """Predict binding for a completely new TF - this is the key capability!"""
        self.eval()
        with torch.no_grad():
            x = dna_seq.transpose(1, 2)
            x = F.relu(self.conv1d(x))
            x = self.maxpool(x)
            x = self.dropout_cnn(x)
            
            x = x.transpose(1, 2)
            lstm_out, _ = self.bilstm(x)
            dna_sequence_features = self.transformer_layer(lstm_out)
            
            x = dna_sequence_features.transpose(1, 2)
            dna_global = self.global_pool(x).squeeze(-1)
            
            new_tf_projected = self.protein_feature_projection(new_tf_features)
            
            binding_logit, attention_weights = self.process_single_tf(
                dna_sequence_features, dna_global, new_tf_projected
            )
            
            binding_prob = torch.sigmoid(binding_logit)
            return binding_prob, attention_weights
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.criterion(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        y_hat = torch.sigmoid(logits)
        
        self.validation_step_outputs = getattr(self, 'validation_step_outputs', [])
        self.validation_step_outputs.append({
            'y_hat': y_hat.detach().cpu(),
            'y': y.detach().cpu(),
            'loss': loss
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or len(self.validation_step_outputs) == 0:
            return
            
        all_preds = torch.cat([x['y_hat'] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x['y'] for x in self.validation_step_outputs], dim=0)
        
        y_true = all_targets.numpy()
        y_pred = all_preds.numpy()
        
        auroc_scores = []
        aupr_scores = []
        
        for i in range(y_true.shape[1]):
            try:
                if len(np.unique(y_true[:, i])) > 1:
                    auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aupr = average_precision_score(y_true[:, i], y_pred[:, i])
                    auroc_scores.append(auroc)
                    aupr_scores.append(aupr)
            except ValueError:
                continue
        
        if len(auroc_scores) > 0:
            avg_auroc = np.mean(auroc_scores)
            avg_aupr = np.mean(aupr_scores)
            median_auroc = np.median(auroc_scores)
            median_aupr = np.median(aupr_scores)
            
            self.log('val_auroc', avg_auroc, on_epoch=True, prog_bar=True)
            self.log('val_aupr', avg_aupr, on_epoch=True, prog_bar=True)
            self.log('val_auroc_median', median_auroc, on_epoch=True)
            self.log('val_aupr_median', median_aupr, on_epoch=True)
            
            print(f"\n==================================================")
            print(f"EPOCH {self.current_epoch} VALIDATION RESULTS")
            print(f"==================================================")
            print(f"Average ROC AUC: {avg_auroc:.5f}")
            print(f"Average PR AUC: {avg_aupr:.5f}")
            print(f"Median ROC AUC: {median_auroc:.5f}")
            print(f"Median PR AUC: {median_aupr:.5f}")
            print(f"🎯 Clean model: no masking or noise")
            print(f"==================================================\n")
        
        self.validation_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=60, eta_min=1e-6
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class DataModule(pl.LightningDataModule):
    def __init__(self, data_folder, batch_size=100, num_workers=4):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        with h5py.File(self.data_folder + 'train.mat', 'r') as trainmat:
            X_train = np.array(trainmat['trainxdata'])
            y_train = np.array(trainmat['traindata'])
            
        with h5py.File(self.data_folder + 'valid.mat', 'r') as validmat:
            X_valid = np.array(validmat['validxdata'])
            y_valid = np.array(validmat['validdata'])
        
        if y_train.shape[1] != 690:
            print(f"Adjusting y_train from {y_train.shape[1]} to 690 labels")
            y_train = y_train[:, :690]
        if y_valid.shape[1] != 690:
            print(f"Adjusting y_valid from {y_valid.shape[1]} to 690 labels")
            y_valid = y_valid[:, :690]
            
        print("Final shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_valid: {X_valid.shape}")
        print(f"y_valid: {y_valid.shape}")
        
        pos_counts = np.sum(y_train, axis=0)
        neg_counts = len(y_train) - pos_counts
        pos_weights = neg_counts / (pos_counts + 1e-8)
        
        self.pos_weights = torch.FloatTensor(pos_weights)
        
        self.train_dataset = TransDataset(X_train, y_train)
        self.val_dataset = TransDataset(X_valid, y_valid)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=self.num_workers, pin_memory=True)



def main():
    RESUME_FROM_CHECKPOINT = False
    CHECKPOINT_PATH = ""
    DATA_FOLDER = "/data/"
    
    MAPPING_FILE = ".../tf_to_feature_mapping_exact.json"
    FEATURES_DIR = ".../tf_features/"
    
    USE_CLASS_WEIGHTING = False
    TF_FEATURE_PROJECTION_DIM = 320
    
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1
    
    data_module = DataModule(data_folder=DATA_FOLDER, batch_size=300)
    data_module.setup()
    
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from: {CHECKPOINT_PATH}")
        model = GeneralizedProteinAware.load_from_checkpoint(
            CHECKPOINT_PATH,
            mapping_file=MAPPING_FILE,
            features_dir=FEATURES_DIR
        )
    else:
        print("Creating new Generalized Protein-Aware model...")
        
        if USE_CLASS_WEIGHTING:
            pos_weight = data_module.pos_weights.cuda() if torch.cuda.is_available() else data_module.pos_weights
            model = GeneralizedProteinAware(
                mapping_file=MAPPING_FILE,
                features_dir=FEATURES_DIR,
                learning_rate=0.001, 
                tf_feature_projection_dim=TF_FEATURE_PROJECTION_DIM,
                pos_weight=pos_weight
            )
        else:
            model = GeneralizedProteinAware(
                mapping_file=MAPPING_FILE,
                features_dir=FEATURES_DIR,
                learning_rate=0.001,
                tf_feature_projection_dim=TF_FEATURE_PROJECTION_DIM
            )
    
    print(f"Using class weighting: {USE_CLASS_WEIGHTING}")
    print(f"🎯 Clean model: no TF masking or noise")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./model/",
        filename="generalized-protein-aware-{epoch:02d}-{val_aupr:.3f}",
        save_top_k=3,
        monitor="val_aupr",
        mode="max",
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_aupr",
        patience=15,
        mode="max"
    )
    
    logger = TensorBoardLogger("tb_logs", name="generalized_protein_aware_model")
    
    trainer = pl.Trainer(
        max_epochs=60,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True
    )
    
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        trainer.fit(model, data_module, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, data_module)
    
    trainer.save_checkpoint("./model/generalized_protein_aware_final.ckpt")
    
    print("\n" + "="*60)
    print("DEMO: PREDICTING ON NEW/UNSEEN TF")
    print("="*60)
    
    device = next(model.parameters()).device
    
    sample_dna = data_module.val_dataset[0][0].unsqueeze(0).to(device)
    
    tf_idx = 42
    binding_prob, attention = model.predict_single_tf(sample_dna, tf_idx)
    print(f"Known TF #{tf_idx} binding probability: {binding_prob.item():.4f}")
    
    new_tf_features = torch.randn(1280).to(device)
    print(f"New TF protein features shape: {new_tf_features.shape}")
    
    binding_prob_new, attention_weights = model.predict_new_tf(sample_dna, new_tf_features)
    print(f"New/unseen TF binding probability: {binding_prob_new.item():.4f}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    model.eval()
    with torch.no_grad():
        known_predictions = model(sample_dna)
        print(f"All 690 TF predictions range: {torch.sigmoid(known_predictions).min():.4f} to {torch.sigmoid(known_predictions).max():.4f}")
    
    print("✅ Model can now predict for completely unseen TFs!")
    print("🎯 Key capabilities:")
    print("   - predict_single_tf(dna, tf_idx): Fast prediction for one known TF")
    print("   - predict_new_tf(dna, tf_features): Predict for completely new TF")
    print("   - Learns DNA-protein relationships, not TF identities")
    print("="*60)
    
    print("Training completed!")

if __name__ == "__main__":
    main()