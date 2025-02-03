# BIDpred

# Installation (In Ubuntu 18.04, 20.04)

```python
pip install "setuptools<58.0" # setuptools setting   
pip install bioservices # wheels setting
```

1. Generate the conda environment
```python
conda env create --file env.yaml --name BID
conda activate BID 
```
 
2. Install related packages  
```python
pip install wget
pip install biopython
pip install biotite
pip install fair-esm  
sudo apt-get install dssp  # for generating RSA using Biopython
```

3. Install models
```python
# at first, it may take some time to download the esm-2, esm-if model  
import torch
_, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D") # load esm-2 model

import esm
_, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50() # load esm-if model
```


# BUG FIX from esm

ImportError: cannot import name 'esmfold_structure_module_only_8M' from 'esm.pretrained' (/home/{user}/anaconda3/envs/Bepitope/lib/python3.8/site-packages/esm/pretrained.py) 

simply copy-paste the functions starting from esmfold_structure_module_only_8M into the pretrained.py
(https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/pretrained.py#L274)

or

cp pretrained.py /home/{user}/anaconda3/envs/Bepitope/lib/python3.8/site-packages/esm/pretrained.py

Then everything will be ok.

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
- filtered_dataset.csv is data collected from SAbDab with filtering cutoff
- epitope_annotation.csv is annotated from Ab-Ag complex with 6 Angstrom distance
- train_csv contains immunodominance annotations of the training set (92 sets)
- test_csv contains immunodominance annotations of the test set (24 sets)


# Prediction


```python
# predict the epitopes from pdb (fetched)
python inference.py --pdb 1cfi
```



```python
# for multiple inference... in bash
for pdb in pdb1 pdb2 pdb3 pdb4 pdb5 pdb6 ... pdb10
> do
> python inference.py --pdb $pdb
> done
```


```python
# for inference using pdb file from local computer...
# the pdb file must be located in Custom_PDB(default) directory 
# or any directory you assign with --pdb_path
python inference_customPDB.py --pdb 6FNZ
```

# Replication

```python
# simply evaluate the trained model(in checkpoint directory) on the epitope3d test set (45 PDB)
python evaluate.py
```

```python
# to train the models and save
# in model training, we recommend using GPU. CPU work is quite slow.
python train.py --model_save_dir models
```


```python
# assign new directory with --model_checkpoint
# python evaluate.py simply evaluate the models saved in checkpoint
python evaluate.py --model_checkpoint models
```

Each python program contains more arguments.
You can check with -h option

ex) python inference.py/inference_CustomPDB.py/train.py/evaluate.py -h
