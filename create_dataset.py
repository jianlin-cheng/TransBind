import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from Bio.PDB import PDBParser
import os
import esm

class ProteinDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.esm_model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.root = root
        self.pdb_files = self.get_pdb_files(os.path.join(root, 'raw'))
        super(ProteinDataset, self).__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    def get_esm_embedding(self, sequence):
        data = [(None, sequence)]  #label and the sequence
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True) #returns a dictionary 
        token_representations = results["representations"][33] #embeddings 
        esm_embedding = token_representations[0, 1:-1].numpy()
        return esm_embedding

    def process_pdb_to_graph(self, structure):
        edges = []
        edge_features = []
        node_features = []
        distance_threshold = 5.0

        sequence = ""
        residues_list = []
        window_size = 2048
        stride = 512

        for model in structure:
            for chain in model:
                residues = list(chain)
                sequence = "".join([res.get_resname() for res in residues if 'CA' in res])

                # Sliding window
                num_windows = (len(sequence) - window_size) // stride + 1
                embeddings = []

                for start in range(0, num_windows * stride, stride):
                    end = min(start + window_size, len(sequence))
                    window_sequence = sequence[start:end]
                    window_embedding = self.get_esm_embedding(window_sequence)

                    # Ensure window_embedding is an array
                    if isinstance(window_embedding, np.ndarray):
                        embeddings.append(window_embedding)
                    else:
                        print(f"Warning: Window embedding is not an array but {type(window_embedding)}")

                # Average embeddings
                if embeddings:
                    combined_embedding = np.mean(np.array(embeddings), axis=0)
                else:
                    # If embeddings are empty, use a default size based on the embedding's first output
                    combined_embedding = np.zeros((window_size, self.esm_model.embed_dim))  # Use embed_dim instead, placeholder value 

                # Mapping combined embeddings to residues
                residue_to_embedding = {}
                index = 0
                for res in residues:
                    if 'CA' in res:
                        if index < len(combined_embedding):
                            residue_to_embedding[index] = combined_embedding[index]
                        else:
                            residue_to_embedding[index] = np.zeros(combined_embedding.shape[1])
                        index += 1

                for i, res1 in enumerate(residues):
                    if 'CA' not in res1:
                        continue
                    ca1 = res1['CA'].get_coord()

                    if i in residue_to_embedding:
                        node_feature = torch.tensor(residue_to_embedding[i])
                        node_feature = torch.cat([node_feature, torch.tensor(ca1)])

                        node_features.append(node_feature.clone())

                        for j in range(i + 1, len(residues)):
                            res2 = residues[j]
                            if 'CA' not in res2:
                                continue
                            ca2 = res2['CA'].get_coord()
                            distance = np.linalg.norm(ca1 - ca2)

                            if distance <= distance_threshold:
                                edges.append([i, j])
                                edges.append([j, i])
                                edge_features.append([distance])
                                edge_features.append([distance])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.stack(node_features)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_pdb_files(self, raw_dir):
        return [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.pdb')]

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.pdb_files]

    @property
    def processed_file_names(self):
        return ['protein_data.pt']

    def process(self):
        data_list = []
        pdb_parser = PDBParser()
        for pdb_file in self.pdb_files:
            structure = pdb_parser.get_structure('Protein', pdb_file)
            data = self.process_pdb_to_graph(structure)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


root = './protein_dataset'
print('Loading dataset...')
dataset = ProteinDataset(root=root)


