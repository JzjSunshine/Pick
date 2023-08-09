#!/bin/bash
python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=5555 \
train.py -c config.json -d 1 --local_world_size 1

# --nnode=0：这个参数指定了节点的数量，这里设置为 0，表示只有一个节点。

# --node_rank=0：这个参数指定了当前节点的排名，这里设置为 0，表示当前节点的排名为 0。

# --nproc_per_node=0：这个参数指定了每个节点上的进程数量，这里设置为 0，表示每个节点上没有其他进程。

# --master_addr=127.0.0.1：这个参数指定了主节点的地址，这里设置为 127.0.0.1，表示主节点在本地运行。

# --master_port=5555：这个参数指定了主节点的端口号，这里设置为 5555。

# python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4 \
# --master_addr=127.0.0.1 --master_port=5555 \
# train.py -c config.json -d 1,2,3,4 --local_world_size 4