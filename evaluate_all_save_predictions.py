import os
import torch
import numpy as np
import pandas as pd
import joblib
from data_provider.data_factory import data_provider_baseline
from models import BatteryJEPA
from sklearn.metrics import mean_absolute_percentage_error
from safetensors.torch import load_file
import time

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_scaler_dir(dataset_name):
    base_dir = "./checkpoints_jepa_finetuned"
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if f"dataset{dataset_name}_" in d or d.endswith(f"dataset{dataset_name}"):
                full_path = os.path.join(base_dir, d)
                if os.path.exists(os.path.join(full_path, 'label_scaler')):
                    return full_path
    fallback_paths = {
        'MIX_large': './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0003_dm64_el2_dl2_datasetMIX_large_lossMAPE_seed2021_alignTrue_headresidual_mlp/',
        'ZN-coin': './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetZN-coin_lossMAPE_seed2021_alignTrue_headresidual_mlp/',
        'NAion': './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetNAion_lossMAPE_seed2021/',
        'CALB': './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetCALB_lossMAPE_seed2021_alignTrue_headresidual_mlp/'
    }
    return fallback_paths.get(dataset_name, None)

def get_configs(dataset_name, align_features, head_type):
    # Set dataset-specific batch sizes to match training hyperparameters exactly
    if dataset_name == 'CALB' or dataset_name == 'NAion':
        batch_size = 8
    elif dataset_name == 'ZN-coin':
        batch_size = 16
    else:
        batch_size = 32
    return Config(
        task_name='long_term_forecast', early_cycle_threshold=100,
        charge_discharge_length=100, seq_len=5, pred_len=5, label_len=48,
        seasonal_patterns='Monthly', enc_in=1, dec_in=1, c_out=1,
        d_model=64, n_heads=4, e_layers=2, d_layers=2, d_ff=128,
        factor=1, dropout=0.1, activation='relu', output_num=1,
        class_num=8, head_type=head_type, head_hidden_dim=256,
        dataset=dataset_name, align_features=align_features,
        root_path='./dataset', data='Dataset_original', loader='modal',
        features='M', num_workers=0, batch_size=batch_size,
        weighted_loss=False, seed=2021
    )

def evaluate_and_save(model_name, dataset_name, checkpoint_dir, align_features, head_type, scaling_name):
    print(f"\n==========================================")
    print(f"Evaluating {model_name} on {dataset_name} ({scaling_name})...")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"==========================================")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found at {checkpoint_dir}. Skipping.")
        return None
        
    configs = get_configs(dataset_name, align_features, head_type)
    
    # Load label_scaler and life_class_scaler
    label_scaler_path = os.path.join(checkpoint_dir, 'label_scaler')
    life_class_scaler_path = os.path.join(checkpoint_dir, 'life_class_scaler')
    
    if not os.path.exists(label_scaler_path) or not os.path.exists(life_class_scaler_path):
        scaler_dir = get_scaler_dir(dataset_name)
        if scaler_dir:
            print(f"Loading scalers from fallback directory: {scaler_dir}")
            label_scaler_path = os.path.join(scaler_dir, 'label_scaler')
            life_class_scaler_path = os.path.join(scaler_dir, 'life_class_scaler')
        else:
            raise FileNotFoundError(f"Scalers not found and no fallback directory available for dataset {dataset_name}")
            
    label_scaler = joblib.load(label_scaler_path)
    life_class_scaler = joblib.load(life_class_scaler_path)
    
    # Load test dataset
    test_data, test_loader = data_provider_baseline(configs, 'test', None, label_scaler, life_class_scaler=life_class_scaler)
    
    # Instantiate Model
    model = BatteryJEPA.Model(configs)
    model.set_mode('downstream')
    
    # Load state dict
    state_dict = load_file(os.path.join(checkpoint_dir, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_preds = []
    total_refs = []
    total_reps = []
    
    std, mean_value = np.sqrt(label_scaler.var_[-1]), label_scaler.mean_[-1]
    
    with torch.no_grad():
        for cycle_curve_data, curve_attn_mask, labels, *_ in test_loader:
            cycle_curve_data = cycle_curve_data.float().to(device)
            curve_attn_mask = curve_attn_mask.float().to(device)
            
            # Forward call returns predictions
            outputs = model(cycle_curve_data, curve_attn_mask)
            
            # Extract latent representations
            # online_encoder(cycle_curve_data, curve_attn_mask) -> [B, L, d_model]
            rep = model.online_encoder(cycle_curve_data, curve_attn_mask)
            
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            
            total_preds.extend(transformed_preds.cpu().numpy().reshape(-1).tolist())
            total_refs.extend(transformed_labels.cpu().numpy().reshape(-1).tolist())
            total_reps.append(rep.cpu().numpy())
            
    total_preds = np.array(total_preds)
    total_refs = np.array(total_refs)
    total_reps = np.concatenate(total_reps, axis=0)  # Shape [N, L, d_model]
    
    # Calculate pooled and reshaped features
    pooled_reps = total_reps.mean(axis=1)            # Shape [N, d_model]
    reshaped_reps = total_reps.reshape(total_reps.shape[0], -1)  # Shape [N, L * d_model]
    
    # Metrics
    mape = mean_absolute_percentage_error(total_refs, total_preds)
    relative_error = np.abs(total_preds - total_refs) / total_refs
    acc_15 = np.mean(relative_error <= 0.15) * 100
    acc_10 = np.mean(relative_error <= 0.10) * 100
    
    print(f"Results: MAPE = {mape:.5f} | Acc@15% = {acc_15:.2f}% | Acc@10% = {acc_10:.2f}%")
    
    # Save predictions and representations
    save_filename = f"reused_eval_{model_name}_{dataset_name}_{scaling_name}.npz"
    save_path = os.path.join("./figure_data", save_filename)
    np.savez_compressed(
        save_path,
        predictions=total_preds,
        targets=total_refs,
        latent_features=total_reps,
        pooled_features=pooled_reps,
        reshaped_features=reshaped_reps
    )
    print(f"✓ Saved prediction data and latent features to: {save_path}")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'scaling': scaling_name,
        'mape': mape,
        'acc_15': acc_15,
        'acc_10': acc_10
    }

def main():
    runs = [
        # Battery-JEPA Aligned
        ('JEPA', 'MIX_large', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0003_dm64_el2_dl2_datasetMIX_large_lossMAPE_seed2021_alignTrue_headresidual_mlp/', True, 'residual_mlp', 'aligned'),
        ('JEPA', 'ZN-coin', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetZN-coin_lossMAPE_seed2021_alignTrue_headresidual_mlp/', True, 'residual_mlp', 'aligned'),
        ('JEPA', 'NAion', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetNAion_lossMAPE_seed2021/', True, 'mlp', 'aligned'),
        ('JEPA', 'CALB', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetCALB_lossMAPE_seed2021_alignTrue_headresidual_mlp/', True, 'residual_mlp', 'aligned'),
        
        # Battery-JEPA Unaligned
        ('JEPA', 'MIX_large', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetMIX_large_lossMAPE_seed2021_alignFalse_headresidual_mlp/', False, 'residual_mlp', 'unaligned'),
        ('JEPA', 'ZN-coin', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0001_dm64_el2_dl2_datasetZN-coin_lossMAPE_seed2021_alignFalse_headresidual_mlp/', False, 'residual_mlp', 'unaligned'),
        ('JEPA', 'NAion', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0003_dm64_el2_dl2_datasetNAion_lossMAPE_seed2021_alignFalse_headresidual_mlp/', False, 'residual_mlp', 'unaligned'),
        ('JEPA', 'CALB', './checkpoints_jepa_finetuned/BatteryJEPA_freezeTrue_tuneinputTrue_lr0.0003_dm64_el2_dl2_datasetCALB_lossMAPE_seed2021_alignFalse_headmlp/', False, 'mlp', 'unaligned'),
        
        # MAE
        ('MAE', 'MIX_large', './checkpoints_jepa/MAE_freezeTrue_tuneinputTrue_datasetMIX_large_seed2021/', True, 'residual_mlp', 'aligned'),
        ('MAE', 'ZN-coin', './checkpoints_jepa/MAE_freezeTrue_tuneinputTrue_datasetZN-coin_seed2021/', False, 'residual_mlp', 'unaligned'),
        ('MAE', 'NAion', './checkpoints_jepa/MAE_freezeTrue_tuneinputTrue_datasetNAion_seed2021/', False, 'residual_mlp', 'unaligned'),
        ('MAE', 'CALB', './checkpoints_jepa/MAE_freezeTrue_tuneinputTrue_datasetCALB_seed2021/', True, 'residual_mlp', 'aligned'),
        
        # SimCLR
        ('SimCLR', 'MIX_large', './checkpoints_jepa/SimCLR_freezeTrue_tuneinputTrue_datasetMIX_large_seed2021/', True, 'residual_mlp', 'aligned'),
        ('SimCLR', 'ZN-coin', './checkpoints_jepa/SimCLR_freezeTrue_tuneinputTrue_datasetZN-coin_seed2021/', False, 'residual_mlp', 'unaligned'),
        ('SimCLR', 'NAion', './checkpoints_jepa/SimCLR_freezeTrue_tuneinputTrue_datasetNAion_seed2021/', False, 'residual_mlp', 'unaligned'),
        ('SimCLR', 'CALB', './checkpoints_jepa/SimCLR_freezeTrue_tuneinputTrue_datasetCALB_seed2021/', True, 'residual_mlp', 'aligned')
    ]
    
    # Expected results from manuscript
    targets = {
        ('JEPA', 'MIX_large', 'aligned'): (0.1731, 64.24),
        ('JEPA', 'MIX_large', 'unaligned'): (0.1793, 59.16),
        ('JEPA', 'ZN-coin', 'unaligned'): (0.3439, 34.74),
        ('JEPA', 'ZN-coin', 'aligned'): (0.4091, 13.96),
        ('JEPA', 'NAion', 'unaligned'): (0.2532, 40.00),
        ('JEPA', 'NAion', 'aligned'): (0.2985, 29.79),
        ('JEPA', 'CALB', 'aligned'): (0.1196, 79.87),
        ('JEPA', 'CALB', 'unaligned'): (0.1880, 40.25),
        
        ('MAE', 'MIX_large', 'aligned'): (0.1949, 62.45),
        ('MAE', 'ZN-coin', 'unaligned'): (0.4150, 14.74),
        ('MAE', 'NAion', 'unaligned'): (0.2771, 31.46),
        ('MAE', 'CALB', 'aligned'): (0.2449, 23.69),
        
        ('SimCLR', 'MIX_large', 'aligned'): (0.2044, 51.71),
        ('SimCLR', 'ZN-coin', 'unaligned'): (0.4003, 19.74),
        ('SimCLR', 'NAion', 'unaligned'): (0.3466, 16.88),
        ('SimCLR', 'CALB', 'aligned'): (0.1195, 78.62),
    }
    
    results = []
    for model_name, dataset_name, checkpoint_dir, align_features, head_type, scaling_name in runs:
        res = evaluate_and_save(model_name, dataset_name, checkpoint_dir, align_features, head_type, scaling_name)
        if res is not None:
            results.append(res)
            
    print("\n" + "="*95)
    print("VERIFICATION TABLE")
    print("="*95)
    print(f"{'Model':<8} | {'Dataset':<10} | {'Scaling':<10} | {'Eval MAPE':<10} | {'Target MAPE':<11} | {'Eval Acc':<9} | {'Target Acc':<10} | {'Status':<6}")
    print("-"*95)
    
    mismatches = 0
    for res in results:
        key = (res['model'], res['dataset'], res['scaling'])
        target_mape, target_acc = targets.get(key, (0.0, 0.0))
        
        # We check within some tolerance (e.g. 0.0002 for MAPE, 0.05 for Acc) to accommodate numerical/device differences
        mape_match = abs(res['mape'] - target_mape) < 0.0002
        acc_match = abs(res['acc_15'] - target_acc) < 0.05
        
        status = "OK" if (mape_match and acc_match) else "MISMATCH"
        if not (mape_match and acc_match):
            mismatches += 1
            
        print(f"{res['model']:<8} | {res['dataset']:<10} | {res['scaling']:<10} | {res['mape']:.5f}    | {target_mape:.5f}     | {res['acc_15']:.2f}%     | {target_acc:.2f}%     | {status:<6}")
        
    print("="*95)
    if mismatches == 0:
        print("✓ SUCCESS: All evaluated models matched the manuscript table results perfectly!")
    else:
        print(f"Warning: {mismatches} configurations did not match the targets within the tolerance.")

if __name__ == '__main__':
    main()
