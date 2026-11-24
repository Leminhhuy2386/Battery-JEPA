import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from accelerate import Accelerator
from data_provider.data_factory import data_provider_baseline
from models import BatteryJEPA
import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_mask(batch_size, seq_len, mask_ratio=0.4, strategy='random', device='cpu'):
    """
    Generates a binary mask of shape [B, L] where 1 represents TARGET (masked) tokens 
    and 0 represents CONTEXT (visible) tokens.
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    num_masked = int(seq_len * mask_ratio)
    num_masked = max(1, min(seq_len - 1, num_masked))
    
    if strategy == 'random':
        for i in range(batch_size):
            indices = torch.randperm(seq_len)[:num_masked]
            mask[i, indices] = 1.0
    elif strategy == 'block':
        # Mask a contiguous block
        block_len = num_masked
        for i in range(batch_size):
            start = torch.randint(0, seq_len - block_len + 1, (1,)).item()
            mask[i, start:start+block_len] = 1.0
    elif strategy == 'future':
        # Mask the last part of the sequence
        start = seq_len - num_masked
        mask[:, start:] = 1.0
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
        
    return mask

def augment_batch(x, jitter_std=0.01, scale_min=0.95, scale_max=1.05):
    if jitter_std <= 0 and scale_min >= 1.0 and scale_max <= 1.0:
        return x
        
    x_augmented = x.clone()
    if jitter_std > 0:
        noise = torch.randn_like(x_augmented) * jitter_std
        x_augmented = x_augmented + noise
        
    if scale_min < 1.0 or scale_max > 1.0:
        B = x_augmented.shape[0]
        scales = torch.empty(B, 1, 1, 1, device=x_augmented.device).uniform_(scale_min, scale_max)
        x_augmented = x_augmented * scales
        
    return x_augmented

def main():
    parser = argparse.ArgumentParser(description='Battery-JEPA Pre-training')
    
    # basic config
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_jepa/', help='location of checkpoints')
    
    # data loader config
    parser.add_argument('--charge_discharge_length', type=int, default=100, help='resampled length for charge/discharge curves')
    parser.add_argument('--dataset', type=str, default='HUST', help='dataset folder for pre-training')
    parser.add_argument('--data', type=str, default='BatteryLife', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/HUST_dataset/', help='root path of the data file')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--num_workers', type=int, default=1, help='dataloader num workers')
    parser.add_argument('--weighted_sampling', action='store_true', default=False, help='use weighted sampling')
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--align_features', action='store_true', help='apply dimensionless cycle/initial scaling to features')
    
    # architecture / training config
    parser.add_argument('--early_cycle_threshold', type=int, default=100, help='max cycles length (sequence length)')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of intra-cycle layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of inter-cycle layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--output_num', type=int, default=1, help='unused downstream output shape')
    
    # optimization
    parser.add_argument('--pretrain_epochs', type=int, default=15, help='number of pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='pretraining learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    
    # JEPA specific config
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio')
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random', 'block', 'future'], help='masking strategy')
    parser.add_argument('--ema_beta', type=float, default=0.996, help='target encoder EMA decay factor')
    
    # Phase 2 Physics & Consistency configs
    parser.add_argument('--mono_weight', type=float, default=0.1, help='weight of latent monotonicity loss')
    parser.add_argument('--mono_margin', type=float, default=0.005, help='margin for monotonicity loss')
    parser.add_argument('--jitter_std', type=float, default=0.01, help='standard deviation for random noise jitter')
    parser.add_argument('--scale_min', type=float, default=0.95, help='min scale factor for capacity/voltage scaling')
    parser.add_argument('--scale_max', type=float, default=1.05, help='max scale factor for capacity/voltage scaling')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator()
    accelerator.print(args.__dict__)
    
    # Initialize Battery-JEPA model
    model = BatteryJEPA.Model(args)
    model.set_mode('pretrain')
    
    # Dataloader setups
    train_data, train_loader = data_provider_baseline(args, 'train', None, sample_weighted=args.weighted_sampling)
    
    # Collect trainable parameters (only online encoder and predictor parameters)
    trainable_params = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.wd)
    
    # Accelerate setups
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    
    if accelerator.is_local_main_process:
        wandb.init(project="Battery-JEPA_Pretraining", config=args.__dict__, mode='offline')
        
    os.makedirs(args.checkpoints, exist_ok=True)
    
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0
        total_jepa_loss = 0
        total_mono_loss = 0
        epoch_time = time.time()
        
        for i, (cycle_curve_data, curve_attn_mask, _, _, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Prepare inputs
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            
            B, L, _, _ = cycle_curve_data.shape
            
            # Generate JEPA Target Mask [B, L]
            target_mask = generate_mask(
                batch_size=B, 
                seq_len=L, 
                mask_ratio=args.mask_ratio, 
                strategy=args.mask_strategy, 
                device=accelerator.device
            )
            
            # Generate Consistency augmented input for Online Encoder
            cycle_curve_data_aug = augment_batch(
                cycle_curve_data, 
                jitter_std=args.jitter_std, 
                scale_min=args.scale_min, 
                scale_max=args.scale_max
            )
            
            # Forward pass through JEPA
            predicted_targets, gt_targets = model(
                cycle_curve_data, 
                curve_attn_mask, 
                target_mask=target_mask,
                cycle_curve_data_aug=cycle_curve_data_aug
            )
            
            # Latent space L2 Loss
            loss_jepa = F.mse_loss(predicted_targets, gt_targets)
            
            # Monotonicity Loss
            unwrapped_model = accelerator.unwrap_model(model)
            loss_mono = unwrapped_model.get_monotonicity_loss(cycle_curve_data, curve_attn_mask, margin=args.mono_margin)
            
            # Combined Loss
            loss = loss_jepa + args.mono_weight * loss_mono
            
            accelerator.backward(loss)
            optimizer.step()
            
            # Momentum update for target encoder
            unwrapped_model.update_target_encoder(args.ema_beta)
            
            total_loss += loss.item()
            total_jepa_loss += loss_jepa.item()
            total_mono_loss += loss_mono.item()
            
            if (i + 1) % 5 == 0:
                accelerator.print(f"Epoch [{epoch+1}/{args.pretrain_epochs}] | Iter [{i+1}/{len(train_loader)}] | Combined Loss: {loss.item():.6f} | JEPA Loss: {loss_jepa.item():.6f} | Mono Loss: {loss_mono.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_jepa = total_jepa_loss / len(train_loader)
        avg_mono = total_mono_loss / len(train_loader)
        epoch_dur = time.time() - epoch_time
        accelerator.print(f"=== Epoch {epoch+1} Complete | Avg Combined Loss: {avg_loss:.6f} | Avg JEPA: {avg_jepa:.6f} | Avg Mono: {avg_mono:.6f} | Duration: {epoch_dur:.2f}s ===")
        
        if accelerator.is_local_main_process:
            wandb.log({
                "epoch": epoch+1, 
                "pretrain_loss": avg_loss,
                "jepa_loss": avg_jepa,
                "mono_loss": avg_mono
            })
            
    # Save the pre-trained checkpoint
    if accelerator.is_local_main_process:
        align_suffix = "_aligned" if getattr(args, 'align_features', False) else "_unaligned"
        checkpoint_path = os.path.join(args.checkpoints, f"jepa_pretrain_{args.dataset}_{args.mask_strategy}{align_suffix}.pth")
        torch.save(accelerator.unwrap_model(model).state_weight_or_dict() if hasattr(accelerator.unwrap_model(model), 'state_weight_or_dict') else accelerator.unwrap_model(model).state_dict(), checkpoint_path)
        print(f"Pre-trained checkpoint saved to: {checkpoint_path}")
        wandb.finish()

if __name__ == '__main__':
    main()
