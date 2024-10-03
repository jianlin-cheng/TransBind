import requests
import os

# Define UniProt API search endpoint
uniprot_url = "https://rest.uniprot.org/uniprotkb/search"

# Modified query to fetch only reviewed (Swiss-Prot) transcription factors in Homo sapiens
query = 'transcription factor AND reviewed:true AND organism_id:9606'  # organism_id:9606 is for humans

# Parameters for the API request
params = {
    'query': query,  # The search query
    'format': 'tsv',  # Tab-separated format for easy parsing
    'fields': 'accession',  # We are fetching only the accession numbers for now
    'size': 100  # Limit results for testing (increase if needed)
}

# Function to fetch UniProt accessions based on the query
def fetch_uniprot_accessions(params):
    accessions = []
    response = requests.get(uniprot_url, params=params)
    
    if response.status_code == 200:
        # Split the result into lines and process each line
        data = response.text.strip().split('\n')
        for line in data[1:]:  # Skip the header row
            accession = line.split('\t')[0]
            accessions.append(accession)
        print(f"Found {len(accessions)} reviewed transcription factors.")
    else:
        print(f"Failed to fetch data from UniProt. Status code: {response.status_code}")
    
    return accessions

# Function to download the FASTA sequence for a given UniProt accession
def download_fasta(accession, output_dir="fasta_files"):
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(fasta_url)
    
    if response.status_code == 200:
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the FASTA sequence to a file
        with open(f"{output_dir}/{accession}.fasta", 'w') as fasta_file:
            fasta_file.write(response.text)
        print(f"Downloaded FASTA for {accession}")
    else:
        print(f"Failed to download FASTA for {accession}. Status code: {response.status_code}")


if __name__ == "__main__":
    # Fetch the list of UniProt accessions
    tf_accessions = fetch_uniprot_accessions(params)
    
    # Download the corresponding FASTA sequences
    for accession in tf_accessions:
        download_fasta(accession)
