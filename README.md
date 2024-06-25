# BIDpred

# Dataset
In cluster_msa_annotation, the dataset for train, test is available
Each name corresponses to (PDB_ID)_(Antigen_chain)_(Antibody_Hchain_Lchain)

Data curation
1. Antigen sequences were clustered using mmseq2 by sequence identity 70%.
2. Within cluster (at least 4 elements), multiple sequence alignment (MSA) was generated using ClustalW
3. Epitope was annotated in the MSA from antigen-antibody complex data (6 Angstrom)

You can read the data using Biopython

'''python
from Bio import AlignIO
align = AlignIO.read(file_path, "phylip")
'''
