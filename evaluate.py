from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn import metrics
from model import GAT
import argparse
import pickle
import torch

arg_parser = argparse.ArgumentParser(description="Model Evaluation...")

arg_parser.add_argument("--model_checkpoint", type=str, default="checkpoint")

arg_parser.add_argument("--kfold", type=int, default=10)

arg_parser.add_argument("--num_head", type=int, default=8)

arg_parser.add_argument("--device", type=str, default="cuda")


args = arg_parser.parse_args()

model_checkpoint = args.model_checkpoint
kfold = args.kfold
num_head = args.num_head
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


with open("test_data.pkl", "rb") as f:
    test_pyg_list = pickle.load(f)


## fixed version
import os
def ensemble(direc, data_pyg_list, kfold):
    model = GAT(in_dim=1792, hid_dim1=2048, hid_dim2=512, out_dim=128, num_head=8, out_head=1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.cuda()
    num_unmasked_nodes = 0 
    for i in data_pyg_list:
        num_unmasked_nodes += len(i.y[i.train_mask])
    pred_ensem = torch.zeros([num_unmasked_nodes])
    
    pt_list = []
    for pt in os.listdir(f"{direc}/"):
        if pt[-3:] == ".pt":
            pt_list.append(pt)
    for pt in pt_list:
        model.load_state_dict(torch.load(f'{direc}/{pt}'))
        model.eval()
        pred_ensem_list = []
        true_label_list = []
        
        pred_label_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_pyg_list):
                batch.to(device)
                out = model(batch)
                y_pred = out[batch.train_mask].reshape(-1)
                for i in y_pred:
                    pred_label_list.append(i)
                for j in batch.y[batch.train_mask]:    
                    true_label_list.append(j.cpu())
            
        pred_label_list = torch.tensor(pred_label_list)
        pred_ensem += pred_label_list
        
    for i in (pred_ensem / kfold): # 10 fold
        pred_ensem_list.append(i.cpu())

    return true_label_list, pred_ensem_list



true_label_list, pred_label_list = ensemble(direc=model_checkpoint, data_pyg_list=test_pyg_list, kfold=kfold)

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

pearson_corr = pearsonr(true_label_list, pred_label_list)
print("Pearson correlation coefficient:", pearson_corr)

mse_value = mean_squared_error(true_label_list, pred_label_list)
print("Root Mean Square Deviation (RMSD):", mse_value ** (1/2))

spearman_corr = spearmanr(true_label_list, pred_label_list)
print("Spearman correlation coefficient:", spearman_corr)

r2 = r2_score(true_label_list, pred_label_list)
print("r2 score:", r2)

print("")
print("="* 50)