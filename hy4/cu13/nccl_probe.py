#!/usr/bin/env python3
"""Run via torchrun on both nodes; real TP8/PP2 NCCL fabric preflight."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def main():
    local=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if torch.cuda.get_device_capability(local)!=(8,0):
        raise RuntimeError("SM80 required")
    dist.init_process_group("nccl",timeout=timedelta(minutes=3))
    try:
        rank=dist.get_rank()
        assert dist.get_world_size()==16 and rank%8==local
        tp=[dist.new_group(list(range(i,i+8))) for i in (0,8)]
        pp=[dist.new_group([i,i+8]) for i in range(8)]
        x=torch.full((4096,),rank,dtype=torch.float32,device="cuda")
        dist.all_reduce(x,group=tp[rank//8])
        assert torch.equal(x,torch.full_like(x,sum(range(rank//8*8,rank//8*8+8))))
        # Real PP payload: flattened iHC BF16 and int32 Top-K, including T>3.
        h=torch.full((1025,4*6144),local+1,dtype=torch.bfloat16,device="cuda")
        top=torch.arange(1025*2048,dtype=torch.int32,device="cuda").reshape(1025,2048)+local
        expected_h=h.clone(); expected_top=top.clone()
        if rank<8:
            dist.send(h,dst=rank+8,group=pp[local])
            dist.send(top,dst=rank+8,group=pp[local])
        else:
            h.zero_(); top.fill_(-1)
            dist.recv(h,src=rank-8,group=pp[local])
            dist.recv(top,src=rank-8,group=pp[local])
            assert torch.equal(h,expected_h) and torch.equal(top,expected_top)
        dist.barrier()
        print("HY4_NCCL_TP_PP=PASS rank",rank,flush=True)
    finally:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
