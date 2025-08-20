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

class TransBindDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence position awareness"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class ProteinFeatureLoader:

    
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
        
        # Load all features
        loaded_count = 0
        for idx, feature_file in enumerate(self.feature_files):
            features = self.load_single_feature_file(feature_file)
            if features is not None:
          
                if len(features) == feature_dim:
                    self.feature_matrix[idx] = features
                    loaded_count += 1
                else:
                    print(f"Dimension mismatch for {feature_file}: expected {feature_dim}, got {len(features)}")
                    # Use zeros for mismatched dimensions
                    self.feature_matrix[idx] = np.zeros(feature_dim)
            else:
                # Use zeros for missing files
                self.feature_matrix[idx] = np.zeros(feature_dim)
        
        print(f"Successfully loaded {loaded_count}/{num_feature_files} feature files")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        # Features are pre-processed, so no normalization needed
        self.feature_matrix = self.normalize_features(self.feature_matrix)
    
    def normalize_features(self, features):
        """Features are already processed, no normalization needed"""
        print("Features are pre-processed, skipping normalization")
        return features
    
    def get_tf_features_matrix(self):
        """Get the TF features matrix aligned with the 690 labels"""
        tf_features = np.zeros((690, self.feature_matrix.shape[1]))
        
        for tf_idx in range(690):
            feature_id = self.tf_to_feature_mapping[tf_idx]
            if feature_id != -1:  # -1 means no mapping
                tf_features[tf_idx] = self.feature_matrix[feature_id]
            # else: keep zeros for unmapped TFs
        
        return tf_features

class ProteinAware_TransBind(pl.LightningModule):
    def __init__(self, mapping_file, features_dir, learning_rate=0.000328, dropout=0.088, 
                 weight_decay=0.028, pos_weight=None, use_cross_attention=True, 
                 tf_feature_projection_dim=320):
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
        
        self.pos_encoding = PositionalEncoding(320, max_len=100)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=320,
            nhead=16,         
            dim_feedforward=1024, 
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
        self.protein_feature_projection = nn.Linear(original_feature_dim, tf_feature_projection_dim)
        self.register_buffer('tf_features_raw', torch.FloatTensor(tf_features_matrix))
        
        # Cross-attention setup
        self.use_cross_attention = use_cross_attention
        
        if self.use_cross_attention:
       
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=320, num_heads=16, batch_first=True 
            )
    
            self.attention_proj = nn.Sequential(
                nn.Linear(320, 320),
                nn.ReLU(),
                nn.Dropout(dropout)  
            )
        if self.use_cross_attention:
            self.fc1 = nn.Linear(640, 1024)  
        else:
            self.fc1 = nn.Linear(640, 1024) 
            
        self.fc2 = nn.Linear(1024, 690) 
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []
    
    def get_projected_tf_features(self):
        """Get protein features projected to the desired dimension"""
        return self.protein_feature_projection(self.tf_features_raw)  # (690, projection_dim)
    
    def forward(self, x):

        x = x.transpose(1, 2) 
        x = F.relu(self.conv1d(x)) 
        x = self.maxpool(x) 
        x = self.dropout_cnn(x)
        x = x.transpose(1, 2) 
        lstm_out, _ = self.bilstm(x)  
        x = self.pos_encoding(lstm_out)
        dna_sequence_features = self.transformer_layer(x) 
        
        # Get global DNA representation
        x = dna_sequence_features.transpose(1, 2)  
        dna_global = self.global_pool(x).squeeze(-1) 
        if self.use_cross_attention:
            # Get projected protein features
            tf_features = self.get_projected_tf_features()  # (690, projection_dim)
            batch_size = dna_global.size(0)
            
            # Expand TF features for batch processing
            tf_batch = tf_features.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 690, projection_dim)
            attended_dna, attention_weights = self.cross_attention(
                query=tf_batch,              
                key=dna_sequence_features,    
                value=dna_sequence_features   
            )
            attended_with_residual = attended_dna + tf_batch  # (batch, 690, 320)
            tf_aware_features = self.attention_proj(attended_with_residual.mean(dim=1))  # (batch, 320)
            combined_features = torch.cat([dna_global, tf_aware_features], dim=1)  # (batch, 640)
            
        else:
      
            tf_features = self.get_projected_tf_features()  # (690, projection_dim)
            avg_tf_features = tf_features.mean(dim=0)  # (projection_dim,)
            tf_context = avg_tf_features.unsqueeze(0).expand(dna_global.size(0), -1)  # (batch, projection_dim)
            combined_features = torch.cat([dna_global, tf_context], dim=1)  # (batch, 640)
        x = F.relu(self.fc1(combined_features))  # Now 640 -> 1024
        x = self.fc2(x)  # 1024 -> 690 raw logits
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  
        loss = self.criterion(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # Raw logits
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
        
        # Calculate average metrics
        if len(auroc_scores) > 0:
            avg_auroc = np.mean(auroc_scores)
            avg_aupr = np.mean(aupr_scores)
            median_auroc = np.median(auroc_scores)
            median_aupr = np.median(aupr_scores)
            
            # Log metrics
            self.log('val_auroc', avg_auroc, on_epoch=True, prog_bar=True)
            self.log('val_aupr', avg_aupr, on_epoch=True, prog_bar=True)
            self.log('val_auroc_median', median_auroc, on_epoch=True)
            self.log('val_aupr_median', median_aupr, on_epoch=True)
            
            # Print detailed results
            print(f"\n==================================================")
            print(f"EPOCH {self.current_epoch} VALIDATION RESULTS")
            print(f"==================================================")
            print(f"Average ROC AUC: {avg_auroc:.5f}")
            print(f"Average PR AUC: {avg_aupr:.5f}")
            print(f"Median ROC AUC: {median_auroc:.5f}")
            print(f"Median PR AUC: {median_aupr:.5f}")
            print(f"Cross-attention: {self.use_cross_attention}")
            print(f"Using OPTIMIZED parameters (16 heads, LR=0.000328)")
            print(f"Using real protein features: True")
            print(f"==================================================\n")
        

        self.validation_step_outputs = []
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=60, eta_min=1e-6
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class TransBindDataModule(pl.LightningDataModule):
    def __init__(self, data_folder, batch_size=200, num_workers=4):
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
        pos_counts = np.sum(y_train, axis=0)  # Count positive samples per class
        neg_counts = len(y_train) - pos_counts  # Count negative samples per class
        pos_weights = neg_counts / (pos_counts + 1e-8)  # Avoid division by zero
        self.pos_weights = torch.FloatTensor(pos_weights)
        
        self.train_dataset = TransBindDataset(X_train, y_train)
        self.val_dataset = TransBindDataset(X_valid, y_valid)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=self.num_workers, pin_memory=True)

def main():
    # Configuration
    RESUME_FROM_CHECKPOINT = False
    CHECKPOINT_PATH = ""
    DATA_FOLDER = "/data/"
    
    # 🆕 NEW: Protein feature configuration
    MAPPING_FILE = ".../tf_to_feature_mapping_exact.json"
    FEATURES_DIR = ".../tf_features/"
    
    # 🚀 OPTIMIZED: Model configuration with ALL optimized parameters
    USE_CROSS_ATTENTION = True
    USE_CLASS_WEIGHTING = False
    TF_FEATURE_PROJECTION_DIM = 320
    
    # 🚀 OPTIMIZED HYPERPARAMETERS (from your best trial: AUPR=0.3700)
    OPTIMIZED_PARAMS = {
        'learning_rate': 0.000328,     
        'dropout': 0.088,             
        'weight_decay': 0.028,       
        'transformer_heads': 16,     
        'transformer_dim_feedforward': 1024, 
        'fc1_size': 1024,           
    }
    
    print("🚀 USING OPTIMIZED HYPERPARAMETERS:")
    for key, value in OPTIMIZED_PARAMS.items():
        print(f"   {key}: {value}")
    
    # Set device
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1
    
    # Data module
    data_module = TransBindDataModule(data_folder=DATA_FOLDER, batch_size=256)
    data_module.setup()  # Setup to get pos_weights
    
    # Model
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from: {CHECKPOINT_PATH}")
        model = ProteinAware_TransBind.load_from_checkpoint(
            CHECKPOINT_PATH,
            mapping_file=MAPPING_FILE,
            features_dir=FEATURES_DIR
        )
    else:
     
        
        if USE_CLASS_WEIGHTING:
            pos_weight = data_module.pos_weights.cuda() if torch.cuda.is_available() else data_module.pos_weights
            model = ProteinAware_TransBind(
                mapping_file=MAPPING_FILE,
                features_dir=FEATURES_DIR,
                learning_rate=OPTIMIZED_PARAMS['learning_rate'],
                dropout=OPTIMIZED_PARAMS['dropout'],
                weight_decay=OPTIMIZED_PARAMS['weight_decay'],
                use_cross_attention=USE_CROSS_ATTENTION,
                tf_feature_projection_dim=TF_FEATURE_PROJECTION_DIM,
                pos_weight=pos_weight
            )
        else:
            model = ProteinAware_TransBind(
                mapping_file=MAPPING_FILE,
                features_dir=FEATURES_DIR,
                learning_rate=OPTIMIZED_PARAMS['learning_rate'],
                dropout=OPTIMIZED_PARAMS['dropout'],
                weight_decay=OPTIMIZED_PARAMS['weight_decay'],
                use_cross_attention=USE_CROSS_ATTENTION,
                tf_feature_projection_dim=TF_FEATURE_PROJECTION_DIM
            )
    

    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./model/",
        filename="protein-aware-{epoch:02d}-{val_aupr:.4f}",
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
    
    # Logger
    logger = TensorBoardLogger("tb_logs", name="OPTIMIZED_protein_aware")
    
    # Trainer
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
    
    # Training
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        trainer.fit(model, data_module, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, data_module)
    
    # Save final model
    trainer.save_checkpoint("./model/OPTIMIZED_protein_aware_final.ckpt")
  

if __name__ == "__main__":
    main()
