import torch
import numpy as np
import scipy.io
import pandas as pd
import argparse
import os
from train_general import GeneralizedProteinAware

def load_tf_features(fea_file):
    with open(fea_file, 'r') as f:
        numbers = [float(x) for x in f.read().split()]
    
    data = np.array(numbers)
    if len(numbers) == 1280:
        return torch.FloatTensor(data)
    elif len(numbers) % 1280 == 0:
        return torch.FloatTensor(data.reshape(-1, 1280).mean(axis=0))
    else:
        return torch.FloatTensor(data[:1280])

def load_sequences(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    if 'testxdata' in mat_data:
        return torch.FloatTensor(mat_data['testxdata'])
    elif 'sequences' in mat_data:
        return torch.FloatTensor(mat_data['sequences'])
    else:
        return torch.FloatTensor(mat_data['data'])

def main():
    parser = argparse.ArgumentParser(description='Predict protein binding probabilities')
    parser.add_argument('--model_path', type=str, default="model/model_general.ckpt",
                       help='Path to the model checkpoint file')
    parser.add_argument('--mapping_file', type=str, default="data/tf_features/tf_to_feature_mapping_exact.json",
                       help='Path to the TF feature mapping JSON file')
    parser.add_argument('--features_dir', type=str, default="data/tf_features",
                       help='Directory containing TF features')
    parser.add_argument('--tf_fea_file', type=str, required=True,
                       help='Path to your TF features file (.fea)')
    parser.add_argument('--sequences_file', type=str, required=True,
                       help='Path to your DNA sequences file (.mat)')
    parser.add_argument('--output_prefix', type=str, default='predictions',
                       help='Prefix for output files (default: predictions)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    required_files = [args.model_path, args.mapping_file, args.tf_fea_file, args.sequences_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    if not os.path.exists(args.features_dir):
        raise FileNotFoundError(f"Features directory not found: {args.features_dir}")
    
    print(f"Loading model from: {args.model_path}")
    print(f"TF features file: {args.tf_fea_file}")
    print(f"Sequences file: {args.sequences_file}")
    
    # Load model
    model = GeneralizedProteinAware.load_from_checkpoint(
        args.model_path, mapping_file=args.mapping_file, features_dir=args.features_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Using device: {device}")
    
    # Load data
    tf_features = load_tf_features(args.tf_fea_file).to(device)
    sequences = load_sequences(args.sequences_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Predict
    predictions = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            prob, _ = model.predict_new_tf(seq.unsqueeze(0).to(device), tf_features)
            predictions.append(prob.item())
            if i % 1000 == 0:
                print(f"Processed {i}/{len(sequences)}")
    
    # Save results
    predictions = np.array(predictions)
    npy_file = f'{args.output_prefix}.npy'
    csv_file = f'{args.output_prefix}.csv'
    
    np.save(npy_file, predictions)
    pd.DataFrame({'binding_probability': predictions}).to_csv(csv_file, index=False)
    
    print(f"Done! {len(predictions)} predictions saved.")
    print(f"Results saved to: {npy_file} and {csv_file}")
    print(f"Mean: {predictions.mean():.4f}, Max: {predictions.max():.4f}")

if __name__ == "__main__":
    main()