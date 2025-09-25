import torch
import numpy as np
import scipy.io
import pandas as pd
from train_general import GeneralizedProteinAware_TransBind

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

# Configuration
MODEL_PATH = "model/model_general.ckpt"
MAPPING_FILE = "data/tf_features/tf_to_feature_mapping_exact.json"
FEATURES_DIR = "data/tf_features"
TF_FEA_FILE = "data/your_tf.fea" #your TF features
SEQUENCES_FILE = "data/sequences.mat" #your DNA sequence

# Load and predict
model = GeneralizedProteinAware_TransBind.load_from_checkpoint(
    MODEL_PATH, mapping_file=MAPPING_FILE, features_dir=FEATURES_DIR
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

tf_features = load_tf_features(TF_FEA_FILE).to(device)
sequences = load_sequences(SEQUENCES_FILE)

predictions = []
with torch.no_grad():
    for i, seq in enumerate(sequences):
        prob, _ = model.predict_new_tf(seq.unsqueeze(0).to(device), tf_features)
        predictions.append(prob.item())
        if i % 1000 == 0:
            print(f"Processed {i}/{len(sequences)}")

# Save results
predictions = np.array(predictions)
np.save('predictions.npy', predictions)
pd.DataFrame({'binding_probability': predictions}).to_csv('predictions.csv', index=False)

print(f"Done! {len(predictions)} predictions saved.")
print(f"Mean: {predictions.mean():.4f}, Max: {predictions.max():.4f}")