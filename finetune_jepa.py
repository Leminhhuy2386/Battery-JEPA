import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from accelerate import Accelerator
from data_provider.data_factory import data_provider_baseline
from models import BatteryJEPA
import wandb
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_baseline, load_content

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Battery-JEPA Downstream Fine-tuning')
    
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_jepa_finetuned/', help='location of checkpoints')
    parser.add_argument('--pretrain_checkpoint', type=str, default='', help='path to pre-trained checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze pre-trained online encoder weights')
    parser.add_argument('--tune_input_projection', action='store_true', help='keep input projection layers trainable under freeze_backbone')
    
    # data loader config
    parser.add_argument('--charge_discharge_length', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='HUST')
    parser.add_argument('--data', type=str, default='BatteryLife')
    parser.add_argument('--root_path', type=str, default='./dataset/HUST_dataset/')
    parser.add_argument('--loader', type=str, default='modal')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--weighted_sampling', action='store_true', default=False)
    parser.add_argument('--align_features', action='store_true', help='apply dimensionless cycle/initial scaling to features')
    
    # forecasting task
    parser.add_argument('--early_cycle_threshold', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=5)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_num', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=8)
    parser.add_argument('--weighted_loss', action='store_true', default=False)
    parser.add_argument('--head_type', type=str, default='mlp', choices=['linear', 'mlp', 'residual_mlp'], help='type of downstream head')
    parser.add_argument('--head_hidden_dim', type=int, default=256, help='hidden dimension for MLP head')
    
    # optimization
    parser.add_argument('--train_epochs', type=int, default=15)
    parser.add_argument('--least_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='MAPE', choices=['MSE', 'MAPE'])
    parser.add_argument('--lradj', type=str, default='constant', choices=['constant', 'COS', 'TST'])
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--input_proj_lr_factor', type=float, default=0.1, help='learning rate multiplier for input projection layers when freeze_backbone is enabled')
    
    # Evaluation alpha-accuracy
    parser.add_argument('--alpha1', type=float, default=0.15)
    parser.add_argument('--alpha2', type=float, default=0.1)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator()
    accelerator.print(args.__dict__)
    
    # Instantiate Model
    model = BatteryJEPA.Model(args)
    model.set_mode('downstream')
    
    # Load pre-trained weights if provided
    if args.pretrain_checkpoint:
        accelerator.print(f"Loading pre-trained checkpoint from: {args.pretrain_checkpoint}")
        checkpoint = torch.load(args.pretrain_checkpoint, map_location='cpu')
        
        # Load weights into the model. Use strict=False since downstream mode uses self.projection
        # which is not trained during pre-training
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        accelerator.print(f"Pre-trained loading completed. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        
    # Apply freeze if requested
    if args.freeze_backbone:
        if args.tune_input_projection:
            accelerator.print("Freezing temporal backbone and PE, but keeping input projection (intra_embed and intra_MLP) trainable...")
            for name, param in model.online_encoder.named_parameters():
                if 'intra_embed' in name or 'intra_MLP' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            accelerator.print("Freezing the entire pre-trained Online Encoder weights...")
            for param in model.online_encoder.parameters():
                param.requires_grad = False
            
    # Load dataset
    train_data, train_loader = data_provider_baseline(args, 'train', None, sample_weighted=args.weighted_sampling)
    label_scaler = train_data.return_label_scaler()
    life_class_scaler = train_data.return_life_class_scaler()
    
    vali_data, vali_loader = data_provider_baseline(args, 'val', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    test_data, test_loader = data_provider_baseline(args, 'test', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    
    # Configure Unique saving path
    setting = 'BatteryJEPA_freeze{}_tuneinput{}_lr{}_dm{}_el{}_dl{}_dataset{}_loss{}_seed{}_align{}_head{}'.format(
        args.freeze_backbone, args.tune_input_projection, args.learning_rate, args.d_model, args.e_layers, args.d_layers, args.dataset, args.loss, args.seed, getattr(args, 'align_features', False), args.head_type
    )
    save_path = os.path.join(args.checkpoints, setting)
    if accelerator.is_local_main_process:
        if os.path.exists(save_path):
            del_files(save_path)
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(label_scaler, f'{save_path}/label_scaler')
        joblib.dump(life_class_scaler, f'{save_path}/life_class_scaler')
        
    accelerator.wait_for_everyone()
    
    # Setup optimizer
    if args.freeze_backbone and args.tune_input_projection:
        head_params = []
        input_proj_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'online_encoder.intra_embed' in name or 'online_encoder.intra_MLP' in name:
                input_proj_params.append(param)
            else:
                head_params.append(param)
        
        optimizer_params = [
            {'params': head_params, 'lr': args.learning_rate},
            {'params': input_proj_params, 'lr': args.learning_rate * args.input_proj_lr_factor}
        ]
        optimizer = optim.Adam(optimizer_params, weight_decay=args.wd)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.wd)
    
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-8)
    elif args.lradj == 'TST':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )
    else:
        scheduler = None
        
    # Prepare accelerator
    train_loader, vali_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, optimizer, scheduler
    )
    
    if accelerator.is_local_main_process:
        wandb.init(project="Battery-JEPA_Downstream", config=args.__dict__, name=setting, mode='offline')
        
    criterion = nn.MSELoss(reduction='none')
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    
    best_vali_loss = float('inf')
    best_test_MAPE = 0
    best_test_alpha_acc1 = 0
    
    for epoch in range(args.train_epochs):
        model.train()
        total_loss = 0
        epoch_time = time.time()
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        
        for i, (cycle_curve_data, curve_attn_mask, labels, _, _, weights, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)
            weights = weights.float().to(accelerator.device)
            
            outputs = model(cycle_curve_data, curve_attn_mask)
            
            if args.loss == 'MSE':
                loss = criterion(outputs, labels)
                loss = torch.mean(loss * weights)
            elif args.loss == 'MAPE':
                # Convert back to raw physical scales for MAPE calculation
                tmp_outputs = outputs * std + mean_value
                tmp_labels = labels * std + mean_value
                loss = criterion(tmp_outputs / tmp_labels, tmp_labels / tmp_labels)
                loss = torch.mean(loss * weights)
                
            accelerator.backward(loss)
            optimizer.step()
            
            if scheduler is not None and args.lradj == 'TST':
                scheduler.step()
                
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation and Testing
        vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(
            args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False
        )
        test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, _, _, _, _ = vali_baseline(
            args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True
        )
        
        if vali_mape < best_vali_loss:
            best_vali_loss = vali_mape
            best_test_MAPE = test_mape
            best_test_alpha_acc1 = test_alpha_acc1
            
        accelerator.print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val MAPE: {vali_mape:.5f} | Test MAPE: {test_mape:.5f} | Test Seen: {test_seen_mape:.5f} | Test Unseen: {test_unseen_mape:.5f}")
        
        if accelerator.is_local_main_process:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_train_loss,
                "val_MAPE": vali_mape,
                "test_MAPE": test_mape,
                "test_seen_MAPE": test_seen_mape,
                "test_unseen_MAPE": test_unseen_mape,
                "test_acc15": test_alpha_acc1
            })
            
        early_stopping(epoch+1, vali_mape, vali_mae_loss, test_mae_loss, model, save_path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping triggered")
            break
            
        if scheduler is not None and args.lradj == 'COS':
            scheduler.step()
            
    accelerator.print(f"=== Optimization Finished ===")
    accelerator.print(f"Best Val Loss: {best_vali_loss:.5f} | Best Test MAPE: {best_test_MAPE:.5f} | Best Test Acc@15%: {best_test_alpha_acc1:.5f}")
    if accelerator.is_local_main_process:
        wandb.finish()

if __name__ == '__main__':
    main()
