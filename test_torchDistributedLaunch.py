import torch
import torch.distributed as dist

import os
import time

print(os.environ)

dist.init_process_group('nccl')

time.sleep(30)

dist.destroy_process_group()

