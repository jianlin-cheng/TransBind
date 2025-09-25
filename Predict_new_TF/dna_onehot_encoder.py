"""Convert DNA sequences to one-hot encoded format and save as .mat file."""

import argparse
import numpy as np
from scipy.io import savemat
import logging

def sequence_to_onehot(sequence, window_size=1000):
    """
    Convert DNA sequence to one-hot encoded format.
    
    Args:
        sequence: DNA sequence string (A, T, G, C, N)
        window_size: Target window size (default: 1000)
    
    Returns:
        numpy array of shape (window_size, 4) where columns are [A, T, G, C]
    """
    # Mapping from nucleotides to indices
    nucleotide_to_index = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    
    # Clean sequence - convert to uppercase
    sequence = sequence.upper().replace('U', 'T')  # Handle RNA sequences
    
    # Initialize one-hot array
    one_hot = np.zeros((window_size, 4), dtype=np.uint8)
    
    # Handle sequence length
    seq_len = len(sequence)
    if seq_len > window_size:
        # Truncate to center portion
        start = (seq_len - window_size) // 2
        sequence = sequence[start:start + window_size]
        seq_len = window_size
    
    # Fill one-hot array
    for i, nucleotide in enumerate(sequence):
        if i >= window_size:
            break
        if nucleotide in nucleotide_to_index:
            one_hot[i, nucleotide_to_index[nucleotide]] = 1
        # Unknown nucleotides (N, etc.) remain as zeros
    
    return one_hot

def get_reverse_complement(sequence):
    """Get reverse complement of DNA sequence."""
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement_map.get(base.upper(), 'N') for base in reversed(sequence))

def process_sequences(sequences, window_size=1000, add_complement=True):
    """
    Process multiple DNA sequences into one-hot format.
    
    Args:
        sequences: List of DNA sequence strings
        window_size: Target window size
        add_complement: Whether to include reverse complement
    
    Returns:
        numpy array of shape (n_samples, window_size, 4)
    """
    all_sequences = []
    
    for seq in sequences:
        # Original sequence
        one_hot = sequence_to_onehot(seq, window_size)
        all_sequences.append(one_hot)
        
        # Add reverse complement if requested
        if add_complement:
            rev_comp = get_reverse_complement(seq)
            one_hot_comp = sequence_to_onehot(rev_comp, window_size)
            all_sequences.append(one_hot_comp)
    
    return np.array(all_sequences)

def load_fasta(filename):
    """Load sequences from FASTA file."""
    sequences = []
    current_seq = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                current_seq = []
            else:
                current_seq.append(line)
        
        # Add last sequence
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences

def save_mat_format(filename, data):
    """Save data in .mat format."""
    if not filename.endswith('.mat'):
        filename += '.mat'
    
    # Create dictionary for .mat file
    mat_data = {
        'data': data,
        'window_size': data.shape[1],
        'n_samples': data.shape[0],
        'channels': 4,  # A, T, G, C
        'channel_names': ['A', 'T', 'G', 'C']
    }
    
    savemat(filename, mat_data)
    print(f"Saved {data.shape[0]} sequences to {filename}")
    print(f"Data shape: {data.shape} (samples, window_size, channels)")

def main():
    parser = argparse.ArgumentParser(description="Convert DNA sequences to one-hot encoded .mat format")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence", help="Single DNA sequence to convert")
    input_group.add_argument("--fasta", help="FASTA file containing sequences")
    input_group.add_argument("--sequences", nargs='+', help="Multiple sequences as command line arguments")
    
    # Options
    parser.add_argument("--output", required=True, help="Output filename (.mat format)")
    parser.add_argument("--window_size", type=int, default=1000, 
                       help="Window size (default: 1000)")
    parser.add_argument("--no_complement", action="store_true",
                       help="Don't include reverse complement sequences")
    
    args = parser.parse_args()
    
    # Get sequences
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.fasta:
        sequences = load_fasta(args.fasta)
        print(f"Loaded {len(sequences)} sequences from {args.fasta}")
    elif args.sequences:
        sequences = args.sequences
    
    if not sequences:
        print("No sequences provided!")
        return
    
    # Process sequences
    add_complement = not args.no_complement
    data = process_sequences(sequences, args.window_size, add_complement)
    
    # Save to .mat file
    save_mat_format(args.output, data)
    
    print(f"\nProcessing complete!")
    print(f"Original sequences: {len(sequences)}")
    print(f"Total samples (with complements): {data.shape[0]}")
    print(f"Window size: {data.shape[1]}")
    print(f"Channels (A,T,G,C): {data.shape[2]}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(__import__('sys').argv) == 1:
        print("Example usage:")
        print("python script.py --sequence 'ATGCGATCG' --output my_sequence.mat")
        print("python script.py --fasta sequences.fasta --output sequences.mat")
        print("python script.py --sequences 'ATGCGATCG' 'CGATCGATC' --output multi_seq.mat")
    else:
        main()