import argparse
import random
import time
import copy
import pickle
import numpy as np

from sklearn.model_selection import KFold
from sklearn import metrics
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch

from model import GAT


arg_parser = argparse.ArgumentParser(description="Model Training...")

arg_parser.add_argument("--epochs", type=int, default=200)

arg_parser.add_argument("--batch_size", type=int, default=4)

arg_parser.add_argument("--seed", type=int, default=76)

arg_parser.add_argument("--kfold", type=int, default=10)


arg_parser.add_argument("--num_head", type=int, default=8)

arg_parser.add_argument("--learning_rate", type=int, default=1e-6)

arg_parser.add_argument("--model_save_dir", type=str, default="checkpoint")

arg_parser.add_argument("--device", type=str, default="cuda")

args = arg_parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
seed = args.seed
kfold = args.kfold
num_head = args.num_head
learning_rate = args.learning_rate
model_save_dir = args.model_save_dir
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print("device", device)


with open("train_data.pkl", "rb") as f:
    train_pyg_list = pickle.load(f)
    
random.seed(seed)
random.shuffle(train_pyg_list)

X = train_pyg_list
kf = KFold(n_splits=kfold) 
kf.get_n_splits(X)
print(kf)

train_start = time.time()

for idx, (train_index, test_index) in enumerate(kf.split(X)):
    
    model = GAT(in_dim=1792, hid_dim1=2048, hid_dim2=512, out_dim=128, num_head=8, out_head=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_val_loss = 100000000
    best_auc_roc = 0
    
    test_sta, test_end = test_index[0], test_index[-1] 
    print("test_index :", test_sta,"~", test_end)
    X_test = X[test_sta:test_end+1]
    print("len X_test :", len(X_test))
    X_train = list(set(X) - set(X_test))
    print("len X_train :", len(X_train))
    
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        
        since = time.time()
        train_loss_list = []
        model.train()

        model.to(device)
        
        for batch in train_loader:
            
            batch.to(device)
            optimizer.zero_grad()
            batch.y = batch.y.float()
            out = model(batch)       
            
            # only surface nodes with (train_mask == True) are used for backpropagation
            loss = loss_fn(out[batch.train_mask].reshape(-1), batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(copy.deepcopy(loss.data.cpu().numpy()))
        epoch_train_avg_loss = np.sum(np.array(train_loss_list))/len(train_loader) 
        
        model.eval()
        val_loss_list = []
        val_r2_list = []
        total_pred_label_list = []
        total_true_label_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(device)
                batch.y = batch.y.float()
                out = model(batch)
                loss = loss_fn(out[batch.train_mask].reshape(-1), batch.y[batch.train_mask])
                
                pred_label_list = []
                for i in out[batch.train_mask]:
                    pred_label_list.append(i.cpu().numpy())
                
                true_label_list = []
                for j in batch.y[batch.train_mask]:
                    true_label_list.append(j.cpu().numpy())
                
            
                val_r2 = r2_score(true_label_list, pred_label_list)
                
                val_r2_list.append(copy.deepcopy(val_r2))
                val_loss_list.append(copy.deepcopy(loss.data.cpu().numpy()))
                   
        epoch_val_avg_r2 = np.sum(np.array(val_r2_list))/len(test_loader) 
        val_r2_history.append(epoch_val_avg_r2)        
        
        
        epoch_val_avg_loss = np.sum(np.array(val_loss_list))/len(test_loader) 
        val_loss_history.append(epoch_val_avg_loss)


        if (epoch_val_avg_loss < best_val_loss) & (epoch_val_avg_r2 > best_val_r2): # 
            best_epoch = epoch+1
            best_val_loss = epoch_val_avg_loss
            best_val_r2 = epoch_val_avg_r2
            best_model_wts = copy.deepcopy(model.state_dict())
            

        end = time.time()
        print(f'{epoch+1}th epoch,')
        print(f'\ttraining loss: {epoch_train_avg_loss:.5f}')
        print(f'\tval loss: {epoch_val_avg_loss:.5f}')
        print(f'\tval r2: {epoch_val_avg_r2:.5f}')
        print(f'\tepoch time: {end-since:.3f}')
            


    save_dir = f'{model_save_dir}'
    import os
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    torch.save(best_model_wts, f'{save_dir}/Best_GAT_{best_epoch}_{idx}.pt')
    print('-'*10)
    print('Train Finished.')
    print(f'Training time: {time.time()-train_start:.2f}s')
    print(f'The best epoch: {best_epoch}')
    print(f'The best val loss: {best_val_loss}')
    print(f'The best val r2: {best_val_r2}')