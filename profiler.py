import time
import torch
from collections import defaultdict
import json
import os

class LayerProfiler:
    def __init__(self, model, save_dir='profile_data'):
        self.model = model
        self.save_dir = save_dir
        self.layer_times = defaultdict(lambda: {'forward': [], 'backward': []})
        self.hooks = []
        self.current_batch = 0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # First, modify all ReLU layers to not use inplace operations
        self._disable_inplace_relu(model)
        
        # Then register hooks
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                self.hooks.extend([
                    module.register_forward_pre_hook(self._forward_pre_hook(name)),
                    module.register_forward_hook(self._forward_hook(name)),
                    # Using register_full_backward_hook instead of register_backward_hook
                    module.register_full_backward_hook(self._backward_hook(name))
                ])
        
        self.start_times = {}
    
    def _disable_inplace_relu(self, model):
        """Recursively modify all ReLU layers to not use inplace operations"""
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(model, child_name, torch.nn.ReLU(inplace=False))
            else:
                self._disable_inplace_relu(child)
    
    def _forward_pre_hook(self, name):
        def hook(module, input):
            torch.cuda.synchronize()
            self.start_times[f"{name}_forward"] = time.perf_counter()
        return hook
    
    def _forward_hook(self, name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            start_time = self.start_times.pop(f"{name}_forward")
            self.layer_times[name]['forward'].append(end_time - start_time)
            return output
        return hook
    
    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            torch.cuda.synchronize()
            if f"{name}_backward" not in self.start_times:
                self.start_times[f"{name}_backward"] = time.perf_counter()
            else:
                end_time = time.perf_counter()
                start_time = self.start_times.pop(f"{name}_backward")
                self.layer_times[name]['backward'].append(end_time - start_time)
            return grad_input  # Full backward hook requires returning grad_input
        return hook
    
    def save_profile_data(self, epoch):
        """Save profiling data for the current epoch"""
        profile_data = {}
        for layer_name, timings in self.layer_times.items():
            profile_data[layer_name] = {
                'forward_avg': sum(timings['forward']) / len(timings['forward']) if timings['forward'] else 0,
                'forward_max': max(timings['forward']) if timings['forward'] else 0,
                'forward_min': min(timings['forward']) if timings['forward'] else 0,
                'backward_avg': sum(timings['backward']) / len(timings['backward']) if timings['backward'] else 0,
                'backward_max': max(timings['backward']) if timings['backward'] else 0,
                'backward_min': min(timings['backward']) if timings['backward'] else 0
            }
        
        filename = os.path.join(self.save_dir, f'profile_epoch_{epoch}.json')
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=4)
        
        self.layer_times.clear()
    
    def close(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()