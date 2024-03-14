import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    def __init__(self, num_task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_task = num_task
        self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.num_task))

    def get_loss_weight(self):
        return torch.exp(-self.loss_scale) * 0.5

    def forward(self, loss, task_index):
        return 0.5 * (torch.exp(-self.loss_scale[task_index]) * loss + self.loss_scale[task_index])


def sample_t_batch(batch_size, clusters, device):
    total_clusters = len(clusters)
    t_list = []
    for min_t, max_t in clusters:
        t = torch.randint(min_t, max_t, (int(batch_size / total_clusters),), device=device)
        t_list.append(t)
    return torch.cat(t_list, dim=0)
