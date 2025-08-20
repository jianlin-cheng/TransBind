#!/usr/bin/env python3
import pandas as pd
import os
import json
import numpy as np
import shutil
from pathlib import Path

class CorrectedCSVTFFeatureMapper:
    def __init__(self, 
                 csv_file="/bml/shreya/BenchMarking_TF/tbinet/create_mapping_label_tf/tf_celltype_list_uniprotID_graphs.csv",
                 feature_index_file="/bml/shreya/BenchMarking_TF/tbinet/data/deepsea_feature_names.txt",
                 deepsea_data_dir="/bml/shreya/BenchMarking_TF/tbinet/data/data_new",
                 final_features_dir="/bml/shreya/BenchMarking_TF/ESM-DBP/output_final"):
        
        self.csv_file = csv_file
        self.feature_index_file = feature_index_file
        self.deepsea_data_dir = deepsea_data_dir
        self.final_features_dir = final_features_dir
        self.tf_features_dir = os.path.join(deepsea_data_dir, "tf_features")
        
        # Create output directory
        os.makedirs(self.tf_features_dir, exist_ok=True)
        
    def load_feature_index_mapping(self):
        """Load the EXACT DeepSEA feature index mapping"""
        print(f"Loading DeepSEA feature index mapping from {self.feature_index_file}")
        
        if not os.path.exists(self.feature_index_file):
            print(f"Error: Feature index file not found: {self.feature_index_file}")
            return None
        
        # Load the feature index file
        df = pd.read_csv(self.feature_index_file, sep='\t')
        
        # Show sample data
        print(f"\nFirst 10 entries:")
        print(df.head(10))
        
        # Create mapping dictionary: feature_name -> index
        feature_to_index = dict(zip(df['Feature_Name'], df['Feature_Index']))
        index_to_feature = dict(zip(df['Feature_Index'], df['Feature_Name']))
        
        print(f"Created mappings for {len(feature_to_index)} features")
        
        return df, feature_to_index, index_to_feature
    
    def load_csv_mapping(self):
        """Load the CSV file with TF mappings"""
        print(f"\nLoading CSV mapping from {self.csv_file}")
        
        if not os.path.exists(self.csv_file):
            print(f"Error: CSV file not found: {self.csv_file}")
            return None
        
        df = pd.read_csv(self.csv_file)
        print(f"Loaded CSV with {len(df)} rows")
        
        return df
    
    def find_matching_feature_file(self, tf_name, uniprot_id):
        """Find the matching .fea file for a given TF name and UniProt ID"""
        # Pattern: TF-name_UniProtID.fea (e.g., AP-2alpha_P05549.fea)
        
        # Special cases mapping
        special_mappings = {
            'Sin3Ak-20': 'SIN3A',     # Sin3Ak-20 -> SIN3A
            'Sin3A-20': 'SIN3A',      # Alternative format
            'Pol2-4H8': 'Pol2',       # RNA Polymerase II -> Pol2
            'PAX5-N19': 'PAX5-C20',   # PAX5-N19 -> PAX5-C20 (exact match found)
            'GATA2-sc267': 'GATA-2',  # GATA2 -> GATA-2 (with hyphen)
            'GATA2': 'GATA-2',        # GATA2 -> GATA-2 (with hyphen, for cases without suffix)
            'eGFP-GATA2': 'GATA-2',   # eGFP-GATA2 -> GATA-2
            'eGFP-FOS': 'c-Fos',      # eGFP-FOS -> c-Fos
            'eGFP-JunD': 'JunD',      # eGFP-JunD -> JunD
            'TCF7L2_C9B9': 'TCF7L2',  # Remove antibody suffix
            'USF1': 'USF-1',          # USF1 -> USF-1 (with hyphen)
            'UBTF': 'UBF',            # UBTF -> UBF
        }
        
        # Check if we have a special mapping
        if tf_name in special_mappings:
            main_tf_name = special_mappings[tf_name]
        else:
            main_tf_name = tf_name
        
        # Create variations of the TF name
        tf_variations = [
            main_tf_name,  # Main/mapped name
            tf_name,  # Original name
            tf_name.replace(' ', '-'),  # Spaces to dashes
            tf_name.replace(' ', '_'),  # Spaces to underscores
            tf_name.replace('-', '_'),  # Dashes to underscores
        ]
        
        # Add simplified versions (remove suffixes after dash/hyphen)
        for variation in tf_variations.copy():
            if '-' in variation:
                # Try removing everything after the last dash
                simplified = variation.split('-')[0]
                tf_variations.append(simplified)
                
                # Also try removing everything after the first dash
                simplified_first = variation.split('-')[0]
                if simplified_first not in tf_variations:
                    tf_variations.append(simplified_first)
            
            # Handle underscore separators too
            if '_' in variation and 'eGFP' not in variation:
                simplified = variation.split('_')[0]
                tf_variations.append(simplified)
        
        # Add uppercase versions
        for variation in tf_variations.copy():
            tf_variations.append(variation.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in tf_variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        # Try all combinations
        possible_filenames = []
        for tf_var in unique_variations:
            possible_filenames.extend([
                f"{tf_var}_{uniprot_id}.fea",
                f"{tf_var.replace(' ', '-')}_{uniprot_id}.fea",
                f"{tf_var.replace(' ', '_')}_{uniprot_id}.fea",
                f"{tf_var.replace('-', '_')}_{uniprot_id}.fea"
            ])
        
        # Remove duplicates
        possible_filenames = list(dict.fromkeys(possible_filenames))
        
        for filename in possible_filenames:
            filepath = os.path.join(self.final_features_dir, filename)
            if os.path.exists(filepath):
                return filename
        
        # Debug: show what we tried for problematic cases
        if any(problem in tf_name for problem in ['Sin3Ak', 'PAX5', 'Pol2', 'GATA2', 'USF1', 'TCF7L2', 'UBTF', 'eGFP']):
            print(f"    DEBUG: Tried {len(possible_filenames)} variations for {tf_name} ({uniprot_id}):")
            for i, filename in enumerate(possible_filenames[:8]):  # Show first 8
                print(f"      {filename}")
            if len(possible_filenames) > 8:
                print(f"      ... and {len(possible_filenames)-8} more")
                
            # Also try to list what files actually exist for this UniProt ID
            existing_files = []
            try:
                import glob
                pattern = os.path.join(self.final_features_dir, f"*{uniprot_id}*.fea")
                existing_files = glob.glob(pattern)
                if existing_files:
                    print(f"    FOUND files with {uniprot_id}:")
                    for f in existing_files:
                        print(f"      {os.path.basename(f)}")
            except:
                pass
        
        return None
    
    def normalize_filename(self, filename):
        """Normalize filename to match between CSV and feature index"""
        # Remove common extensions and variations
        normalized = filename
        normalized = normalized.replace('.narrowPeak.gz', '')
        normalized = normalized.replace('.narrowPeak', '')
        normalized = normalized.replace('UniPk', 'UniPk')  # Ensure consistent ending
        
        return normalized
    
    def create_exact_mappings(self, csv_df, feature_df, feature_to_index):
        """Create EXACT mappings using the feature index file"""
        print(f"\nCreating exact mappings...")
        
        # Create CSV filename to metadata mapping
        csv_mapping = {}
        for _, row in csv_df.iterrows():
            normalized_filename = self.normalize_filename(row['Filename'])
            csv_mapping[normalized_filename] = {
                'tf_name': row['Transcription Factor'],
                'cell_type': row['Cell Type'],
                'uniprot_id': row['UniProt ID'],
                'original_graph_file': row['TF_graph_file']  # Keep original for reference
            }
        
        # Create the exact 690-dimensional mapping
        tf_labels = ['UNKNOWN'] * 690  # Initialize with placeholders
        tf_to_feature_mapping = [-1] * 690  # -1 indicates no mapping
        encode_mappings = {}
        unique_features = set()
        
        matched_count = 0
        unmatched_features = []
        feature_not_found = []
        
        # Go through each DeepSEA feature IN ORDER
        for _, row in feature_df.iterrows():
            feature_index = row['Feature_Index']
            feature_name = row['Feature_Name']
            
            # Normalize the feature name to match CSV
            normalized_feature = self.normalize_filename(feature_name)
            
            if normalized_feature in csv_mapping:
                # Found a match in CSV!
                csv_info = csv_mapping[normalized_feature]
                
                # Now find the corresponding .fea file
                feature_file = self.find_matching_feature_file(
                    csv_info['tf_name'], 
                    csv_info['uniprot_id']
                )
                
                if feature_file:
                    # Found matching feature file!
                    tf_labels[feature_index] = feature_name
                    unique_features.add(feature_file)
                    
                    # Store mapping information
                    encode_mappings[feature_name] = {
                        'index': feature_index,
                        'tf_name': csv_info['tf_name'],
                        'cell_type': csv_info['cell_type'],
                        'uniprot_id': csv_info['uniprot_id'],
                        'feature_file': feature_file
                    }
                    
                    matched_count += 1
                    
                    if matched_count <= 10:  # Show first 10 matches
                        print(f"  {feature_index:3d}: {feature_name} -> {csv_info['tf_name']} -> {feature_file}")
                else:
                    # CSV match but no feature file found
                    feature_not_found.append((feature_index, feature_name, csv_info))
                    if len(feature_not_found) <= 5:  # Show first 5
                        print(f"  {feature_index:3d}: {feature_name} -> {csv_info['tf_name']} -> FEATURE FILE NOT FOUND")
            else:
                unmatched_features.append((feature_index, feature_name))
                if len(unmatched_features) <= 5:  # Show first 5 unmatched
                    print(f"  {feature_index:3d}: {feature_name} -> NO CSV MATCH")
        
        print(f" Matched {matched_count} out of {len(feature_df)} DeepSEA features")
        print(f" {len(unmatched_features)} features have no CSV mapping")
        print(f"{len(feature_not_found)} features have CSV mapping but no feature file")
        
        # Create feature name to ID mapping for matched features
        unique_features_list = sorted(list(unique_features))
        feature_name_to_id = {feature_name: idx for idx, feature_name in enumerate(unique_features_list)}
        
        # Create final feature mapping
        for feature_name, info in encode_mappings.items():
            feature_index = info['index']
            feature_file = info['feature_file']
            if feature_file in feature_name_to_id:
                tf_to_feature_mapping[feature_index] = feature_name_to_id[feature_file]
        
        print(f" Created mappings to {len(unique_features_list)} unique feature files")
        
        return tf_labels, tf_to_feature_mapping, unique_features_list, encode_mappings, unmatched_features, feature_not_found
    
    def organize_feature_files(self, unique_features):
        """Copy and organize the feature files"""
        print(f"\nOrganizing feature files...")
        
        copied_files = []
        missing_files = []
        
        for feature_id, feature_filename in enumerate(unique_features):
            # Source file path
            source_path = os.path.join(self.final_features_dir, feature_filename)
            
            # Target file path with standardized naming
            target_filename = f"feature_{feature_id:03d}.fea"
            target_path = os.path.join(self.tf_features_dir, target_filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                copied_files.append((feature_filename, target_filename))
                
                if feature_id < 5:  # Show first 5
                    print(f"  Copied: {feature_filename} -> {target_filename}")
            else:
                missing_files.append(feature_filename)
                print(f"  Missing: {source_path}")
        
        print(f"Copied {len(copied_files)} feature files")
        if missing_files:
            print(f"Missing {len(missing_files)} feature files")
        
        return copied_files, missing_files
    
    def save_mapping_file(self, tf_labels, tf_to_feature_mapping, unique_features, encode_mappings, unmatched_features, feature_not_found):
        """Save the complete mapping to JSON"""
        print(f"\nSaving mapping file...")
        
        mapping_data = {
            "description": "EXACT DeepSEA TF binding prediction mapping (690 labels to protein features)",
            "source_info": {
                "csv_file": self.csv_file,
                "feature_index_file": self.feature_index_file,
                "deepsea_data": self.deepsea_data_dir,
                "features_source": self.final_features_dir
            },
            
            "dimensions": {
                "num_tf_labels": 690,
                "num_matched_labels": len([x for x in tf_labels if x != 'UNKNOWN']),
                "num_unmatched_labels": len([x for x in tf_labels if x == 'UNKNOWN']),
                "num_unique_features": len(unique_features),
                "unique_features_used": len([x for x in tf_to_feature_mapping if x != -1])
            },
            
            "tf_to_feature_mapping": tf_to_feature_mapping,
            
            "tf_metadata": {
                "tf_labels": tf_labels,
                "tf_ids": list(range(690))
            },
            
            "feature_metadata": {
                "feature_files": [f"feature_{i:03d}.fea" for i in range(len(unique_features))],
                "original_feature_names": unique_features,
                "feature_ids": list(range(len(unique_features)))
            },
            
            "encode_mappings": encode_mappings,
            
            "unmatched_features": [
                {"index": idx, "feature_name": name} 
                for idx, name in unmatched_features
            ],
            
            "feature_files_not_found": [
                {"index": idx, "feature_name": name, "tf_info": info}
                for idx, name, info in feature_not_found
            ],
            
            "statistics": {
                "matched_experiments": len(encode_mappings),
                "unmatched_experiments": len(unmatched_features),
                "missing_feature_files": len(feature_not_found),
                "coverage_percentage": f"{len(encode_mappings)/690*100:.1f}%",
                "labels_per_feature": {
                    str(feature_id): tf_to_feature_mapping.count(feature_id)
                    for feature_id in set(tf_to_feature_mapping) if feature_id != -1
                }
            }
        }
        
        # Save mapping file
        mapping_file = os.path.join(self.deepsea_data_dir, "tf_to_feature_mapping_exact.json")
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f" Saved mapping file: {mapping_file}")
        
        # Save detailed CSV for inspection
        csv_mapping_file = os.path.join(self.deepsea_data_dir, "tf_label_feature_mapping_exact.csv")
        mapping_df = pd.DataFrame({
            'Feature_Index': range(690),
            'ENCODE_Label': tf_labels,
            'Feature_ID': tf_to_feature_mapping,
            'Feature_File': [f"feature_{fid:03d}.fea" if fid != -1 else "NO_FEATURE" for fid in tf_to_feature_mapping],
            'Original_Feature': [unique_features[fid] if fid != -1 else "NO_FEATURE" for fid in tf_to_feature_mapping],
            'Has_Mapping': [label != 'UNKNOWN' for label in tf_labels]
        })
        mapping_df.to_csv(csv_mapping_file, index=False)
        print(f" Saved CSV mapping: {csv_mapping_file}")
        
        return mapping_file
    
    def run_complete_setup(self):
        """Run the complete setup process"""
      
        
        # Step 1: Load feature index mapping (the EXACT order)
        feature_data = self.load_feature_index_mapping()
        if feature_data is None:
            return False
        feature_df, feature_to_index, index_to_feature = feature_data
        
        # Step 2: Load CSV mapping
        csv_df = self.load_csv_mapping()
        if csv_df is None:
            return False
        
        # Step 3: Create exact mappings
        results = self.create_exact_mappings(csv_df, feature_df, feature_to_index)
        tf_labels, tf_to_feature_mapping, unique_features, encode_mappings, unmatched_features, feature_not_found = results
        
        # Step 4: Organize feature files
        copied_files, missing_files = self.organize_feature_files(unique_features)
        
        # Step 5: Save mapping file
        mapping_file = self.save_mapping_file(tf_labels, tf_to_feature_mapping, unique_features, encode_mappings, unmatched_features, feature_not_found)
        
        # Step 6: Print summary
       
        matched_count = len([x for x in tf_labels if x != 'UNKNOWN'])
        print(f"Mapped {matched_count}/690 DeepSEA labels to protein features")
        print(f"Using {len(unique_features)} unique feature files")
        print(f"Coverage: {matched_count/690*100:.1f}%")
        
        if unmatched_features:
            print(f"\nUNMAPPED FEATURES ({len(unmatched_features)}):")
            for idx, name in unmatched_features[:10]:  # Show first 10
                print(f"  {idx:3d}: {name}")
            if len(unmatched_features) > 10:
                print(f"  ... and {len(unmatched_features)-10} more")
        
        if feature_not_found:
            print(f"\n FEATURE FILES NOT FOUND ({len(feature_not_found)}):")
            for idx, name, info in feature_not_found[:10]:  # Show first 10
                print(f"  {idx:3d}: {name} -> {info['tf_name']} ({info['uniprot_id']})")
            if len(feature_not_found) > 10:
                print(f"  ... and {len(feature_not_found)-10} more")
        
        return True

def main():
    """Main function"""
    mapper = CorrectedCSVTFFeatureMapper()
    success = mapper.run_complete_setup()
    if not success:
        print("\nSetup failed. Please check the errors above.")

if __name__ == "__main__":
    main()