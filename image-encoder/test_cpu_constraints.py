"""
Test script to verify CPU constraint configuration

This script demonstrates how the image pipeline configures CPU threading
constraints when CUDA is not available.
"""

import torch
import multiprocessing
import os

print("="*60)
print("SYSTEM INFORMATION")
print("="*60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CPU Cores: {multiprocessing.cpu_count()}")
print(f"PyTorch Version: {torch.__version__}")
print("="*60)

# Import and test the pipeline
from image_pipeline import configure_cpu_constraints

print("\nTesting configure_cpu_constraints()...")
config = configure_cpu_constraints()

print("\nConfiguration Result:")
for key, value in config.items():
    print(f"  {key}: {value}")

print("\nEnvironment Variables:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")

print("\nPyTorch Thread Settings:")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
