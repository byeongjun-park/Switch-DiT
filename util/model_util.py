import torch
from scipy.optimize import linear_sum_assignment

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@torch.jit.script
def compute_gating(k: int, zeros: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size

@torch.no_grad()
def bipartite_matching(x1, x2):
    C = torch.abs(x1[:, :, None] - x2[:, None]).sum(0)
    indices = linear_sum_assignment(C.cpu())
    x2_index = torch.as_tensor(indices[1], dtype=torch.int64)
    return x2_index