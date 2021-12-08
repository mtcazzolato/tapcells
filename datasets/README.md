# Datasets  

This folder provides the cell position of every set of images employed in the experimental evaluation.  

## Image Source  

The sequences of images are from: http://celltrackingchallenge.net/.  

## Detection of cell positions  

The provided files contain the cell positions detected by the TWANG approach. See more details in:  

>  Stegmaier J, Otte JC, Kobitski A, Bartschat A, Garcia A, Nienhaus GU, et al. (2014) Fast Segmentation of Stained Nuclei in Terabyte-Scale, Time Resolved 3D Microscopy Image Stacks. PLoS ONE 9(2): e90036. https://doi.org/10.1371/journal.pone.0090036  

## File organization  

 - Files *"filenames_" + dataset_name + ".txt"* have the references for the list of cells detected at each timestamp (i.e., each image of the sequence).
 - Each folder corresponds to a dataset employed in our experimental evaluation.
 - The list of cells detected at each timestamp (or image) of the sequence is within the corresponding folder (regarding every dataset), in the corresponding file: *"tXXX_TwangSegmentation_RegionProps.csv"*, where *XXX* corresponds to the timestamp (or the number of the image in the sequence).  
