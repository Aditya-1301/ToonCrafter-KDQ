import torch

def debug_tensor(name, tensor, detailed=False):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item()
        }
        print(f"\n[DEBUG] {name} stats:", stats)
        if detailed:
            print(f"Shape: {tensor.shape}")
            print(f"Device: {tensor.device}")
            print(f"Requires grad: {tensor.requires_grad}")
        # import pdb; pdb.set_trace()
    return tensor