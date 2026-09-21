import os
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

# Global torch device used by agent, memory, and networks.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Light cleanup at startup to reduce stale CUDA allocator usage.
gc.collect()
torch.cuda.empty_cache()

# Synchronous CUDA launches help pinpoint stack traces during debugging.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'