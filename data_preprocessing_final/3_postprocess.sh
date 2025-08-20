#!/usr/bin/sh

export PATH=$PATH:$(pwd)/bedtools # bedtools software path
HOME_DIR=/bml/shreya/TF_binding_site/dataset_test/DEEPSEA_dataextraction
data_root=$HOME_DIR/data/processed/

echo Merging...
cat "$data_root"peaks/* > "$data_root"merged.gz

echo Sorting...
bedtools sort -i "$data_root"merged.gz > "$data_root"merged.sorted

echo Gzipping...
gzip "$data_root"merged.sorted -f

rm -rf "$data_root"merged.gz
#rm -rf "$data_root"peaks