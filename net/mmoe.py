import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense_GatingNetwork(nn.Module):
    """ Task-specific gating network for expert selection """
    def __init__(self, input_dim, num_experts):
        super(Dense_GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))  # Shape: (batch, seq_len, num_experts)
    
class Sparse_GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Sparse_GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.fc(x)  # Get raw scores for each expert
        probs = F.softmax(logits, dim=-1)
        indices = torch.argmax(probs, dim=-1, keepdim=True)  # Pick the best expert (Top-1)
        mask = torch.zeros_like(logits).scatter_(-1, indices, 1)  # Create a one-hot mask
        return mask, probs  # Hard routing (only one expert is active)
    
class ExpertNetwork(nn.Module):
    """ Expert network with input 25600 and output 1024 """
    def __init__(self, input_dim=25600, hidden_dim=4096, output_dim=1024):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Reduce to intermediate hidden size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Reduce to final output size (1024)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first reduction
        return self.fc2(x)  # Output compressed feature

class MMoE(nn.Module):
    """ MMoE module with full 25600 input and 1024 output per expert """
    def __init__(self, gate_type, num_experts=5, k=1, feature_dim=1024, num_layers=25, num_tasks=2, hidden_dim=4096, reduced_dim=1024):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim

        # **Per-Layer LayerNorm**
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])

        # **Expert Networks (Now output 1024 instead of 25600)**
        self.experts = nn.ModuleList([ExpertNetwork(feature_dim * num_layers, hidden_dim, reduced_dim) for _ in range(num_experts)])

        # **Task-Specific Gating Networks**
        if gate_type == 'Dense_GatingNetwork':
            self.gates = nn.ModuleList([Dense_GatingNetwork(feature_dim * num_layers, num_experts) for _ in range(num_tasks)])
        elif gate_type == 'Sparse_GatingNetwork':
            self.gates = nn.ModuleList([Sparse_GatingNetwork(feature_dim * num_layers, num_experts) for _ in range(num_tasks)])
        else:
            print("Invalid Gate Type")
        
    def forward(self, layer_reps):
        """
        Forward pass for MMoE with full 25600 input to 1024 output experts.
        
        Args:
            layer_reps (list of torch.Tensor): A list of 25 layer representations from WavLM.
        
        Returns:
            tuple: (se_features, ser_features)  # Shape: (batch, seq_len, 1024)
        """
        # **Apply LayerNorm separately to each WavLM layer**
        normed_layers = [self.layer_norms[i](layer_reps[i]) for i in range(self.num_layers)]

        # **Concatenate normalized layers along feature dimension**
        x = torch.cat(normed_layers, dim=-1)  # Shape: (batch, seq_len, 25600)

        # **Pass through Experts (25600 → 1024)**
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (batch, seq_len, 1024, num_experts)

        task_outputs = []
        gate_probs_list = []
        for gate in self.gates:
            if isinstance(gate, Sparse_GatingNetwork):
                gate_mask, gate_probs = gate(x)  # both (B, T, E)
                gate_weights = gate_mask.unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(gate_probs)  # (B, T, E) → keep for LB loss
            else:
                gate_weights = gate(x).unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(None)  # Not applicable

            task_output = (expert_outputs * gate_weights).sum(dim=-1)  # (B, T, 1024)
            task_outputs.append(task_output)

        return task_outputs, gate_probs_list

class WS_MMoE(nn.Module):
    """ MMoE module with full 25600 input and 1024 output per expert """
    def __init__(self, gate_type, num_experts=5, k=1, feature_dim=1024, num_layers=25, num_tasks=2, hidden_dim=4096, reduced_dim=1024):
        super(WS_MMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim

        # **Per-Layer LayerNorm**
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])

        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.experts = nn.ModuleList([ExpertNetwork(feature_dim, hidden_dim, reduced_dim) for _ in range(num_experts)])

        # **Task-Specific Gating Networks**
        if gate_type == 'Dense_GatingNetwork':
            self.gates = nn.ModuleList([Dense_GatingNetwork(feature_dim , num_experts) for _ in range(num_tasks)])
        elif gate_type == 'Sparse_GatingNetwork':
            self.gates = nn.ModuleList([Sparse_GatingNetwork(feature_dim, num_experts) for _ in range(num_tasks)])
        else:
            print("Invalid Gate Type")
        
    def forward(self, layer_reps):
        """
        Forward pass for MMoE with full 25*1024 input to 1024 output experts (with weighted sum).
        
        Args:
            layer_reps (list of torch.Tensor): A list of 25 layer representations from WavLM.
        
        Returns:
            tuple: (se_features, ser_features)  # Shape: (batch, seq_len, 1024)
        """
        # **Apply LayerNorm separately to each WavLM layer**
        normed_layers = [self.layer_norms[i](layer_reps[i]) for i in range(self.num_layers)]

        # **Concatenate normalized layers along feature dimension**
        stacked = torch.stack(normed_layers, dim=0)
        
        norm_weights = F.softmax(self.layer_weights, dim=0)  # (25,)
        norm_weights = norm_weights.view(self.num_layers, 1, 1, 1)

        # Weighted sum: (B, T, D)
        x = torch.sum(norm_weights * stacked, dim=0)

        # **Pass through Experts (1024 → 1024)**
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (batch, seq_len, 1024, num_experts)

        task_outputs = []
        gate_probs_list = []
        for gate in self.gates:
            if isinstance(gate, Sparse_GatingNetwork):
                gate_mask, gate_probs = gate(x)  # both (B, T, E)
                gate_weights = gate_mask.unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(gate_probs)  # (B, T, E) → keep for LB loss
            else:
                gate_weights = gate(x).unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(None)  # Not applicable

            task_output = (expert_outputs * gate_weights).sum(dim=-1)  # (B, T, 1024)
            task_outputs.append(task_output)

        return task_outputs, gate_probs_list

class WSExpert(nn.Module):
    """Each expert performs a weighted sum over the 25 WavLM layers"""
    def __init__(self, num_layers=25, feature_dim=1024):
        super(WSExpert, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))  # (25,)
        self.num_layers = num_layers
        self.feature_dim = feature_dim

    def forward(self, layer_reps):
        """
        layer_reps: List of 25 tensors, each (B, T, D)
        Returns: (B, T, D)
        """
        stacked = torch.stack(layer_reps, dim=0)  # (25, B, T, D)
        
        norm_weights = F.softmax(self.weights, dim=0).view(self.num_layers, 1, 1, 1)  # (25, 1, 1, 1)
        fused = torch.sum(norm_weights * stacked, dim=0)  # (B, T, D)
        return fused


class Multi_Weighted_Sum(nn.Module):
    """ MMoE module with full 25600 input and 1024 output per expert """
    def __init__(self, gate_type, num_experts=5, k=1, feature_dim=1024, num_layers=25, num_tasks=2, hidden_dim=4096, reduced_dim=1024):
        super(Multi_Weighted_Sum, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim

        # **Per-Layer LayerNorm**
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])

        # **Expert Networks (Now output 1024 instead of 25600)**
        self.experts = nn.ModuleList([WSExpert() for _ in range(num_experts)])

        # **Task-Specific Gating Networks**
        if gate_type == 'Dense_GatingNetwork':
            self.gates = nn.ModuleList([Dense_GatingNetwork(feature_dim * num_layers, num_experts) for _ in range(num_tasks)])
        elif gate_type == 'Sparse_GatingNetwork':
            self.gates = nn.ModuleList([Sparse_GatingNetwork(feature_dim * num_layers, num_experts) for _ in range(num_tasks)])
        else:
            print("Invalid Gate Type")
        
    def forward(self, layer_reps):
        """
        Forward pass for MMoE with full 25600 input to 1024 output experts.
        
        Args:
            layer_reps (list of torch.Tensor): A list of 25 layer representations from WavLM.
        
        Returns:
            tuple: (se_features, ser_features)  # Shape: (batch, seq_len, 1024)
        """
        # **Apply LayerNorm separately to each WavLM layer**
        normed_layers = [self.layer_norms[i](layer_reps[i]) for i in range(self.num_layers)]

        # **Concatenate normalized layers along feature dimension**
        x = torch.cat(normed_layers, dim=-1)  # Shape: (batch, seq_len, 25600)

        # **Pass through Experts (25600 → 1024)**
        expert_outputs = torch.stack([expert(normed_layers) for expert in self.experts], dim=-1)  # (batch, seq_len, 1024, num_experts)

        task_outputs = []
        gate_probs_list = []
        for gate in self.gates:
            if isinstance(gate, Sparse_GatingNetwork):
                gate_mask, gate_probs = gate(x)  # both (B, T, E)
                gate_weights = gate_mask.unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(gate_probs)  # (B, T, E) → keep for LB loss
            else:
                gate_weights = gate(x).unsqueeze(2)  # (B, T, 1, E)
                gate_probs_list.append(None)  # Not applicable

            task_output = (expert_outputs * gate_weights).sum(dim=-1)  # (B, T, 1024)
            task_outputs.append(task_output)

        return task_outputs, gate_probs_list