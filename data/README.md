### Data
To run the MOSTA experiments and reproduce the results from the paper, download the MOSTA data from  https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/Cell_bin_matrix/Mouse_brain_Adult_GEM_CellBin.tsv.gz,
extract `SS200000135TL_D1_CellBin.tsv` and copy it to this folder.
Now call `preprocess_MOSTA.py` to create the preprocessed dataset. The data is saved in a compressed numpy format, containing the feature matrix, the corresponding spatial coordinates of the cells and the names of the selected genes.
