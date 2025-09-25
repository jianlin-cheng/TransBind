import os
import numpy as np
import h5py
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# Import your model class from the training script
from train import ProteinAware_TransBind  #

# --- Step 1: Load the trained model ---


CHECKPOINT_PATH = ""
MAPPING_FILE = "../data/tf_to_feature_mapping_exact.json"
FEATURES_DIR = "../data/tf_features/"

print("Loading trained Protein-Aware model...")
model = ProteinAware_TransBind.load_from_checkpoint(
    CHECKPOINT_PATH,
    mapping_file=MAPPING_FILE,
    features_dir=FEATURES_DIR
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"Model loaded on device: {device}")
print(f"Using cross-attention: {model.use_cross_attention}")
print(f"TF feature projection dimension: {model.tf_feature_projection_dim}")

# --- Step 2: Load test data ---
print("Loading test data...")
with h5py.File('/bml/shreya/BenchMarking_TF/Bioinfor-DeepSEA/data/tfrecords/test.mat', 'r') as testmat:
    X_test = np.array(testmat['testxdata'])  # Shape: (N, 1000, 4)
    y_test = np.array(testmat['testdata'])[:, :690]  # Ensure 690 labels

print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Convert to PyTorch tensor and move to device
X_test_tensor = torch.FloatTensor(X_test).to(device)

# --- Step 3: Make predictions ---
print("Making predictions...")
batch_size = 100
preds = []

with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i + batch_size]
        # Get raw logits and apply sigmoid to get probabilities
        batch_logits = model(batch)
        batch_pred = torch.sigmoid(batch_logits).cpu().numpy()
        preds.append(batch_pred)
        
        # Print progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch)}/{len(X_test_tensor)} samples...")

pred_y = np.concatenate(preds, axis=0)
print(f"Predictions shape: {pred_y.shape}")

# --- Step 4: Evaluate metrics ---
print("Calculating metrics...")
roc_scores, pr_scores = [], []

# Create output filename with model info
output_filename = f'aucs_protein_aware_{"cross_attn" if model.use_cross_attention else "simple"}.txt'

with open(output_filename, 'w') as aucs_file:
    aucs_file.write('Protein-Aware AU ROC\tProtein-Aware AU PR\n')
    
    for i in range(690):
        try:
            # Check if we have both positive and negative samples
            if len(np.unique(y_test[:, i])) > 1:
                roc_auc = roc_auc_score(y_test[:, i], pred_y[:, i])
                pr_auc = average_precision_score(y_test[:, i], pred_y[:, i])
                roc_scores.append(roc_auc)
                pr_scores.append(pr_auc)
                aucs_file.write(f'{roc_auc:.5f}\t{pr_auc:.5f}\n')
            else:
                # All samples are the same class (usually all negative)
                roc_scores.append(np.nan)
                pr_scores.append(np.nan)
                aucs_file.write('NaN\tNaN\n')
        except ValueError as e:
            print(f"Error calculating metrics for label {i}: {e}")
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
            aucs_file.write('NaN\tNaN\n')
    
    # Calculate and write averages
    avg_roc = np.nanmean(roc_scores)
    avg_pr = np.nanmean(pr_scores)
    median_roc = np.nanmedian(roc_scores)
    median_pr = np.nanmedian(pr_scores)
    
    aucs_file.write(f'\nAVERAGE\t{avg_roc:.5f}\t{avg_pr:.5f}\n')
    aucs_file.write(f'MEDIAN\t{median_roc:.5f}\t{median_pr:.5f}\n')
    
    # Count valid scores
    valid_roc_count = np.sum(~np.isnan(roc_scores))
    valid_pr_count = np.sum(~np.isnan(pr_scores))
    aucs_file.write(f'VALID_SCORES\t{valid_roc_count}/690\t{valid_pr_count}/690\n')

# Print summary
print('\n' + '='*60)
print('PROTEIN-AWARE PERFORMANCE SUMMARY')
print('='*60)
print(f'Model configuration:')
print(f'  - Cross-attention: {model.use_cross_attention}')
print(f'  - TF feature projection dim: {model.tf_feature_projection_dim}')
print(f'  - Using real protein features: True')
print(f'')
print(f'Results:')
print(f'  - Average ROC AUC: {avg_roc:.5f}')
print(f'  - Average PR AUC: {avg_pr:.5f}')
print(f'  - Median ROC AUC: {median_roc:.5f}')
print(f'  - Median PR AUC: {median_pr:.5f}')
print(f'  - Valid ROC scores: {valid_roc_count}/690')
print(f'  - Valid PR scores: {valid_pr_count}/690')
print(f'')
print(f'Results saved to: {output_filename}')
print('='*60)

# Additional analysis: Show distribution of scores
print('\nScore distribution:')
print(f'ROC AUC - Min: {np.nanmin(roc_scores):.3f}, Max: {np.nanmax(roc_scores):.3f}')
print(f'PR AUC - Min: {np.nanmin(pr_scores):.3f}, Max: {np.nanmax(pr_scores):.3f}')

# Show top performing labels
if valid_roc_count > 0:
    top_indices = np.argsort(roc_scores)[-10:][::-1]  # Top 10 ROC scores
    print(f'\nTop 10 TF labels by ROC AUC:')
    for idx in top_indices:
        if not np.isnan(roc_scores[idx]):
            print(f'  Label {idx}: ROC={roc_scores[idx]:.3f}, PR={pr_scores[idx]:.3f}')

print('\nTesting completed!')

