#!/bin/bash
export PATH=$PATH:$(pwd)/bedtools # bedtools software path

# Base home directory
HOME_DIR=/bml/shreya/TF_binding_site/dataset_test/DEEPSEA_dataextraction

# inputs
inp_peak_files_dir=$HOME_DIR/data/downloads/wgEncodeAwgTfbsUniform/ # wgEncodeAwgTfbsUniform_small or wgEncodeAwgTfbsUniform
inp_human_genome_file=$HOME_DIR/data/downloads/hg19_latest/hg19.genome 

# outputs
out_dir=$HOME_DIR/data/processed/
out_peak_files_dir="$out_dir"wgEncodeAwgTfbsUniform_sorted/ # wgEncodeAwgTfbsUniform_sorted_small or wgEncodeAwgTfbsUniform_sorted
human_genome_filename="$(basename ${inp_human_genome_file})"
out_human_genome_file=$out_dir$human_genome_filename.windowed.sorted.gz

mkdir -p $out_peak_files_dir


#-------------------------Preprocessing narrowPeak files
echo "Sorting all the peak file in the '$inp_peak_files_dir'..."
if [ -z "$(ls -A $out_peak_files_dir)" ]; then
    for filepath in $inp_peak_files_dir*; do 
        if [[ $filepath == *.narrowPeak.gz ]]; then
            echo -e "\t$filepath";
            filename="$(basename ${filepath} .narrowPeak.gz)";
            # echo $filename;
            bedtools sort -i $filepath > $out_peak_files_dir$filename.narrowPeak.sorted
            gzip $out_peak_files_dir$filename.narrowPeak.sorted -f
        fi
    done
else
   echo -e "\t'$out_peak_files_dir'" contains data. Doing no preprocessing of narrowPeak files.
fi



#-------------------------Preprocessing human genome
window_size=200
step_size=$window_size

if [ -f $out_human_genome_file ]; then
    echo -e "\t'$out_human_genome_file' exists."
else 
    echo "Dividing human genome $window_size-bp bins with step-size $step_size..."  # window_size==step_size indicates no-overlapping
    bedtools makewindows -g $inp_human_genome_file -w $window_size -s $step_size > $out_dir$human_genome_filename.windowed #data/processed/hg19.genome.windowed
    bedtools sort -i $out_dir$human_genome_filename.windowed > $out_dir$human_genome_filename.windowed.sorted   # sorting 200-bp bins
    gzip $out_dir$human_genome_filename.windowed.sorted -f    # gzipping
    rm -rf $out_dir$human_genome_filename.windowed    # removing extra file
fi

