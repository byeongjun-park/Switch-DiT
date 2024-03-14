import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel_experts import ParallelExperts
from util.model_util import compute_gating

class TaskMoE(nn.Module):
    def __init__(self,  hidden_size, num_experts, k):

        super(TaskMoE, self).__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k

        self.f_gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, num_experts, bias=True))
        self.experts = ParallelExperts(num_experts, hidden_size, hidden_size)
        self.active_task = 0

    def top_k_gating(self, task_full):
        logits = self.f_gate(task_full)
        probs = torch.softmax(logits, dim=1)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

        zeros_quantized = torch.zeros((task_full.size(0), self.num_experts), dtype=torch.float32, device=logits.device)
        probs_full_quantized = zeros_quantized.scatter(1, top_k_indices, 1)

        active_probs = probs[self.active_task]
        active_top_k_gates = top_k_gates[self.active_task]
        active_top_k_indices = top_k_indices[self.active_task]

        batch_gates, batch_index, expert_size = (
            compute_gating(self.k, torch.zeros_like(active_probs), active_top_k_gates, active_top_k_indices)
        )

        self.expert_size = expert_size
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        return active_probs, probs_full_quantized

    def set_active_task(self, active_task):
        self.active_task = active_task

    def forward(self, x, task_full):
        batch_size, length, emb_size = x.size()
        probs, probs_full_quantized = self.top_k_gating(task_full)
        expert_inputs = x[self.batch_index].reshape(-1, emb_size)
        expert_outputs = self.experts(expert_inputs, self.expert_size * length)
        expert_outputs = expert_outputs.view(-1, length, emb_size) * self.batch_gates[:, None, None]
        zeros = torch.zeros((batch_size, length, emb_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        out = zeros.index_add(0, self.batch_index, expert_outputs)
        return 1 + out, probs, probs_full_quantized