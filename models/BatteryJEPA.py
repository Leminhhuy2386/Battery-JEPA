import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out


class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim, d_model, d_ff, d_layers, dropout):
        super(ResidualMLPHead, self).__init__()
        self.first_linear = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            MLPBlock(d_model, d_ff, d_model, dropout)
            for _ in range(d_layers)
        ])
        self.out_linear = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.first_linear(x)
        x = F.relu(x)
        for block in self.blocks:
            x = block(x)
        return self.out_linear(F.relu(x))


class BaseEncoder(nn.Module):
    """
    Sub-component representing the common encoder architecture:
    Intra-cycle feature extractor + Positional Embedding + Inter-cycle Transformer.
    """
    def __init__(self, configs):
        super(BaseEncoder, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers

        # Intra-cycle feature extractor
        self.intra_flatten = nn.Flatten(start_dim=2)
        # 3 channels (Voltage, Current, Capacity)
        self.intra_embed = nn.Linear(self.charge_discharge_length * 3, self.d_model)
        self.intra_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) 
            for _ in range(configs.e_layers)
        ])

        # Inter-cycle temporal transformer
        self.pe = PositionalEmbedding(self.d_model)
        self.inter_TransformerEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ]
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, cycle_curve_data, curve_attn_mask=None):
        """
        cycle_curve_data: [B, num_cycles, fixed_len, num_var]
        curve_attn_mask: [B, num_cycles] or None
        """
        B, L, T, C = cycle_curve_data.shape
        x = self.intra_flatten(cycle_curve_data)  # [B, L, T * C]
        x = self.intra_embed(x)  # [B, L, d_model]
        for i in range(self.e_layers):
            x = self.intra_MLP[i](x)  # [B, L, d_model]

        x = self.pe(x) + x

        if curve_attn_mask is not None:
            # Replicate CPTransformer attention mask structure
            attn_mask = curve_attn_mask.unsqueeze(1)  # [B, 1, L]
            attn_mask = torch.repeat_interleave(attn_mask, attn_mask.shape[-1], dim=1)  # [B, L, L]
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L, L]
            attn_mask = (attn_mask == 0)  # Convert to boolean mask where True means masked
        else:
            attn_mask = None

        output, attns = self.inter_TransformerEncoder(x, attn_mask=attn_mask)
        output = self.dropout(output)
        return output  # [B, L, d_model]


class Predictor(nn.Module):
    """
    Predicts representations of target cycles based on context representations
    and target positions.
    """
    def __init__(self, configs):
        super(Predictor, self).__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        
        # We model the predictor as a series of attention blocks mapping context keys/values 
        # to target query locations.
        self.query_projection = nn.Embedding(configs.early_cycle_threshold, self.d_model)
        self.predictor_layers = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ]
        )
        self.proj_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, context_rep, context_mask, target_indices):
        """
        context_rep: [B, L_c, d_model] - Online encoder outputs for visible context
        context_mask: [B, L_c] - Mask showing active context items
        target_indices: [B, L_t] - Integer sequence of targets we want to predict
        """
        B, L_t = target_indices.shape
        # Embed the target indices to get initial queries
        queries = self.query_projection(target_indices)  # [B, L_t, d_model]

        # Combine context and queries for cross-attention or standard attention in a joint encoder.
        # Here we perform cross-attention/encoding by concatenating queries and context:
        # queries are appended at the end of context.
        # Input to predictor: [B, L_c + L_t, d_model]
        combined = torch.cat([context_rep, queries], dim=1)
        
        # Run predictor layers
        out, _ = self.predictor_layers(combined)
        
        # Slice target predictions out
        predicted_targets = out[:, -L_t:, :]  # [B, L_t, d_model]
        predicted_targets = self.proj_out(predicted_targets)
        return predicted_targets


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.d_model = configs.d_model
        
        # Encoders
        self.online_encoder = BaseEncoder(configs)
        self.target_encoder = BaseEncoder(configs)
        
        # Initialize target encoder parameters from online encoder and disable gradient
        self.init_target_encoder()
        
        # Predictor
        self.predictor = Predictor(configs)
        
        # Downstream classification / regression head
        head_type = getattr(configs, 'head_type', 'linear')
        in_dim = self.d_model * self.early_cycle_threshold
        if head_type == 'mlp':
            hidden_dim = getattr(configs, 'head_hidden_dim', 256)
            self.projection = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(hidden_dim // 2, configs.output_num)
            )
        elif head_type == 'residual_mlp':
            d_ff = configs.d_ff
            d_layers = configs.d_layers
            self.projection = ResidualMLPHead(in_dim, self.d_model, d_ff, d_layers, configs.dropout)
        else:
            self.projection = nn.Linear(in_dim, configs.output_num)
        
        self.mode = 'pretrain'  # 'pretrain' or 'downstream'

    def init_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, beta):
        """
        Momentum update of target encoder: ϕ_t = β * ϕ_t + (1 - β) * θ_t
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = beta * param_t.data + (1 - beta) * param_o.data

    def set_mode(self, mode):
        assert mode in ['pretrain', 'downstream']
        self.mode = mode

    def get_monotonicity_loss(self, cycle_curve_data, curve_attn_mask, margin=0.005):
        """
        Computes the physics-guided latent monotonicity loss:
        Penalizes if the distance of cycle t from cycle 1 is not monotonically increasing.
        """
        online_rep = self.online_encoder(cycle_curve_data, curve_attn_mask) # [B, L, d_model]
        B, L, D = online_rep.shape
        if L <= 1:
            return torch.tensor(0.0, device=cycle_curve_data.device)
            
        z1 = online_rep[:, 0:1, :] # [B, 1, D]
        dists = torch.norm(online_rep - z1, p=2, dim=-1) # [B, L]
        diffs = dists[:, :-1] - dists[:, 1:] # [B, L-1]
        loss = F.relu(diffs + margin)
        
        if curve_attn_mask is not None:
            diff_mask = curve_attn_mask[:, :-1] * curve_attn_mask[:, 1:]
            loss = loss * diff_mask
            denom = diff_mask.sum()
            if denom > 0:
                return loss.sum() / denom
        return loss.mean()

    def forward(self, cycle_curve_data, curve_attn_mask, target_mask=None, return_embedding=False, cycle_curve_data_aug=None):
        """
        If self.mode == 'pretrain':
            target_mask: [B, L] - binary mask where 1 indicates TARGET/MASKED cycles, 0 indicates CONTEXT/VISIBLE cycles.
        If self.mode == 'downstream':
            Predicts the downstream target directly.
        """
        if self.mode == 'pretrain':
            assert target_mask is not None, "target_mask is required for pre-training"
            B, L, T, C = cycle_curve_data.shape
            
            # Divide inputs into Context and Target
            context_attn_mask = curve_attn_mask * (1.0 - target_mask)  # Keep only visible context
            online_input = cycle_curve_data_aug if cycle_curve_data_aug is not None else cycle_curve_data
            online_rep = self.online_encoder(online_input, context_attn_mask) # [B, L, d_model]
            
            # Stop gradient on Target Encoder path
            with torch.no_grad():
                target_rep = self.target_encoder(cycle_curve_data, curve_attn_mask) # [B, L, d_model]
                target_rep = target_rep.detach()

            # Gather target indices
            device = cycle_curve_data.device
            batch_target_indices = []
            for b in range(B):
                t_idx = torch.where(target_mask[b] == 1)[0]
                batch_target_indices.append(t_idx)
            
            batch_target_indices = torch.stack(batch_target_indices, dim=0).to(device)  # [B, L_t]
            
            predicted_targets = self.predictor(online_rep, context_attn_mask, batch_target_indices) # [B, L_t, d_model]
            
            # Extract ground-truth target representations
            gt_targets = []
            for b in range(B):
                gt_targets.append(target_rep[b, batch_target_indices[b]])
            gt_targets = torch.stack(gt_targets, dim=0)
            
            return predicted_targets, gt_targets

        else:
            # Downstream evaluation mode
            online_rep = self.online_encoder(cycle_curve_data, curve_attn_mask)  # [B, L, d_model]
            output = online_rep.reshape(online_rep.shape[0], -1)  # [B, L * d_model]
            preds = self.projection(output)  # [B, output_num]
            if return_embedding:
                return preds, output
            return preds
