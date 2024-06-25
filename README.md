# BIDpred

# Dataset
In **cluster_msa_annotation**, the dataset for train, test is available
Each name corresponses to (PDB_ID)_(Antigen_chain)_(Antibody_Hchain_Lchain).phy
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
print(align[0].seq)  # sequence of the first sequence in the alignment
```
In **Rep_Antigen_PDB**, you can get the PDB file of Representative sequence in each MSA.
