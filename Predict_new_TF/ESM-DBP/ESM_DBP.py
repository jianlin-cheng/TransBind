import esm
import torch
import numpy as np
import os
import sys

esm_model_dict_dir = sys.argv[1]
fasta_path = sys.argv[2]
result_dir = sys.argv[3]
device = sys.argv[4]

# Load ESM model
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = torch.nn.DataParallel(esm_model)
esm_model.load_state_dict(torch.load(esm_model_dict_dir + os.sep + 'ESM-DBP.model', map_location=lambda storage, loc: storage))
esm_model.to(device)
esm_model.eval()

def get_one_protein_esm_fea(protein_name, seq):
    print("Generate feature representation of {0}...".format(protein_name))
    data = [(protein_name, seq)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Extract per-residue representations (on device)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        # batch*seq_len*fea_dim
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = torch.squeeze(results["representations"][33])
        return token_representations[1:-1]  # Remove start and end tokens

def readfastaAndSeq(file_path):
    fi = open(file_path, 'r')
    dicts = {}
    current_header = None
    current_seq = ""
    
    for line in fi:
        line = line.strip()
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_header is not None:
                dicts[current_header] = current_seq
            # Start new sequence
            current_header = line[1:]  # Remove '>'
            current_seq = ""
        else:
            # Add to current sequence (concatenate all sequence lines)
            current_seq += line
    
    # Don't forget the last sequence
    if current_header is not None:
        dicts[current_header] = current_seq
    
    fi.close()
    return dicts

def generate_features_only(pro_name, seq):
    with torch.no_grad():
        fea_represent = get_one_protein_esm_fea(pro_name, seq)
        np_fea = fea_represent.clone().to('cpu').detach().numpy()
        np.savetxt(result_dir + os.sep + pro_name + '.fea', np_fea, fmt='%.10f')
        print(f"Feature file saved: {result_dir + os.sep + pro_name + '.fea'}")

if __name__ == '__main__':
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    pro_dicts = readfastaAndSeq(fasta_path)
    
    print(f"Processing {len(pro_dicts)} proteins...")
    for pro, seq in pro_dicts.items():
        generate_features_only(pro, seq)
    
    print('Feature generation complete! Happy Every Day!')