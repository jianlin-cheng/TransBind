import os
import requests

# Function to extract UniProt ID from FASTA header
def extract_uniprot_id(fasta_header):
    try:
        parts = fasta_header.split('|')
        if len(parts) >= 2:
            return parts[1]  # The UniProt ID is the second part
        else:
            raise ValueError("The FASTA header format is not as expected.")
    except Exception as e:
        print(f"Error extracting UniProt ID: {e}")
        return None

# Function to process FASTA file and extract UniProt ID
def process_fasta_file(filepath):
    try:
        with open(filepath, 'r') as file:
            header = file.readline().strip()  # Read the header line
            if header.startswith('>'):
                return extract_uniprot_id(header)
            else:
                print(f"The file {filepath} does not appear to be in FASTA format.")
    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file {filepath}: {e}")
    return None

# Function to download PDB file from AlphaFold
def download_pdb_from_alphafold(uniprot_id, output_dir):
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(pdb_url)
    if response.status_code == 200:
        pdb_filename = os.path.join(output_dir, f"{uniprot_id}.pdb")
        with open(pdb_filename, 'wb') as file:
            file.write(response.content)
        print(f"PDB file for UniProt ID {uniprot_id} downloaded successfully.")
        return pdb_filename
    else:
        print(f"Error: Unable to fetch PDB file for UniProt ID {uniprot_id}. Response Code: {response.status_code}")
        return None

# Function to download FASTA file from UniProt
def download_fasta_from_uniprot(uniprot_id, output_dir):
    fasta_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(fasta_url)
    if response.status_code == 200:
        fasta_filename = os.path.join(output_dir, f"{uniprot_id}.fasta")
        with open(fasta_filename, 'wb') as file:
            file.write(response.content)
        print(f"FASTA file for UniProt ID {uniprot_id} downloaded successfully.")
        return fasta_filename
    else:
        print(f"Error: Unable to fetch FASTA file for UniProt ID {uniprot_id}. Response Code: {response.status_code}")
        return None

# Main function to process all FASTA files and download PDB/FASTA files
def main():
    fasta_dir = 'fasta_files'  # Directory containing FASTA files
    output_dir = './protein_dataset/raw'  # Directory to save PDB/FASTA files
    
    if not os.path.exists(fasta_dir):
        print(f"Directory {fasta_dir} does not exist.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
    
    uniprot_ids = []
    
    try:
        for filename in os.listdir(fasta_dir):
            if filename.endswith('.fasta') or filename.endswith('.fa'):
                filepath = os.path.join(fasta_dir, filename)
                uniprot_id = process_fasta_file(filepath)
                if uniprot_id:
                    uniprot_ids.append(uniprot_id)
        
        # Download PDB and FASTA files for each UniProt ID
        if uniprot_ids:
            for uniprot_id in uniprot_ids:
                download_pdb_from_alphafold(uniprot_id, output_dir)
                #download_fasta_from_uniprot(uniprot_id, output_dir)
                #print('pdb downloaded')
        else:
            print("No UniProt IDs were extracted from the FASTA files.")
    
    except Exception as e:
        print(f"An error occurred while processing the directory {fasta_dir}: {e}")

#main script
if __name__ == "__main__":
    main()
