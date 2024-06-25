# BIDpred


# Dataset
In **cluster_msa_annotation**, the dataset for train, test is available
Each file name corresponds to (PDB_ID)_(Antigen_chain)_(Antibody_Hchain_Lchain).phy
example. 1eo8_A_HL.phy, 3pnw_R_QP.phy

Data curation
1. Antigen sequences were clustered using mmseq2 by sequence identity 70%.
2. Within cluster (at least 4 elements), multiple sequence alignment (MSA) was generated using ClustalW
3. Epitopes were annotated in the MSA from antigen-antibody complex data (6 Angstrom)

*Representative sequence is on the first row </br>
*Epitopes are annotated as capital letter while non-epitopes are not.

You can read the data using Biopython

```python
from Bio import AlignIO
align = AlignIO.read(file_path, "phylip")

print(align[0].id)  # id of the first sequence in the alignment
print(align[0].seq)  # amino acid of the first sequence in the alignment
```
In **Rep_Antigen_PDB**, you can get the PDB file of Representative sequence in each MSA.
Each file name corresponds to (PDB_ID)_(Antigen_chain).pdb

In **csv**, you can get the 
1. filtered_dataset.csv </br>
- filtered_dataset.csv is Data retrieved from SAbDab with filtering cutoff
2. epitope_annotation.csv</br>
- epitope_annotation.csv is annotated from Ab-Ag complex with 6 Angstrom distance
3. train_csv</br>
- train_csv contains immunodominance annotations of the training set (92 sets)
4. test_csv</br>
- test_csv contains immunodominance annotations of the test set (24 sets)
