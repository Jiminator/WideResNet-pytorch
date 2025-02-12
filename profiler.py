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
        self.layer_memory = defaultdict(lambda: {'forward': [], 'backward': []})
        self.hooks = []
        self.current_batch = 0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Reset CUDA memory stats at start
        torch.cuda.reset_peak_memory_stats()
        
        # First, modify all ReLU layers to not use inplace operations
        self._disable_inplace_relu(model)
        
        # Then register hooks
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                self.hooks.extend([
                    module.register_forward_pre_hook(self._forward_pre_hook(name)),
                    module.register_forward_hook(self._forward_hook(name)),
                    module.register_full_backward_pre_hook(self._full_backward_pre_hook(name)),
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
            # Reset peak memory stats before forward pass
            torch.cuda.reset_peak_memory_stats()
        return hook
    
    def _forward_hook(self, name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            # Record timing
            end_time = time.perf_counter()
            start_time = self.start_times.pop(f"{name}_forward")
            self.layer_times[name]['forward'].append(end_time - start_time)
            
            # Record memory usage
            memory_used = torch.cuda.max_memory_reserved() / (1024 * 1024)  # Convert to MB
            self.layer_memory[name]['forward'].append(memory_used)
            return output
        return hook
    
    def _full_backward_pre_hook(self, name):
        def hook(module, grad_output):
            torch.cuda.synchronize()
            self.start_times[f"{name}_backward"] = time.perf_counter()
            # Reset peak memory stats before backward pass
            torch.cuda.reset_peak_memory_stats()
        return hook
    
    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            torch.cuda.synchronize()
            # Record timing
            end_time = time.perf_counter()
            start_time = self.start_times.pop(f"{name}_backward")
            self.layer_times[name]['backward'].append(end_time - start_time)
            
            # Record memory usage
            memory_used = torch.cuda.max_memory_reserved() / (1024 * 1024)  # Convert to MB
            self.layer_memory[name]['backward'].append(memory_used)
            return grad_input
        return hook
    
    def save_profile_data(self, epoch):
        """Save profiling data for the current epoch"""
        profile_data = {}
        for layer_name in self.layer_times.keys():
            timings = self.layer_times[layer_name]
            memory = self.layer_memory[layer_name]
            
            profile_data[layer_name] = {
                'forward_time': {
                    'avg': sum(timings['forward']) / len(timings['forward']) if timings['forward'] else 0,
                    'max': max(timings['forward']) if timings['forward'] else 0,
                    'min': min(timings['forward']) if timings['forward'] else 0
                },
                'backward_time': {
                    'avg': sum(timings['backward']) / len(timings['backward']) if timings['backward'] else 0,
                    'max': max(timings['backward']) if timings['backward'] else 0,
                    'min': min(timings['backward']) if timings['backward'] else 0
                },
                'forward_memory_mb': {
                    'avg': sum(memory['forward']) / len(memory['forward']) if memory['forward'] else 0,
                    'max': max(memory['forward']) if memory['forward'] else 0,
                    'min': min(memory['forward']) if memory['forward'] else 0
                },
                'backward_memory_mb': {
                    'avg': sum(memory['backward']) / len(memory['backward']) if memory['backward'] else 0,
                    'max': max(memory['backward']) if memory['backward'] else 0,
                    'min': min(memory['backward']) if memory['backward'] else 0
                }
            }
        
        filename = os.path.join(self.save_dir, f'profile_epoch_{epoch}.json')
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=4)
        
        # Clear the data for next epoch
        self.layer_times.clear()
        self.layer_memory.clear()
    
    def close(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()