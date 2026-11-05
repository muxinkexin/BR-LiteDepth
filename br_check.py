"""Temporary root-level CPU checks; no KITTI/real weights are needed."""
import os,sys
if __name__ == '__main__' and ('LD_LIBRARY_PATH' in os.environ or 'LD_PRELOAD' in os.environ):
    env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    os.execve(sys.executable,[sys.executable]+sys.argv,env)
import argparse,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import networks
from networks.depth_encoder import CDilated,BRConditionalDepthwise
from datasets.mono_dataset import MonoDataset
from layers import br_cross_scale,br_masked_mean,br_erode,SSIM,BackprojectDepth,Project3D

def fake_k(h,w):
    return torch.tensor([[.58*w,0,.5*w,0],[0,1.92*h,.5*h,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32)

class FakeDataset(MonoDataset):
    def __init__(self,mode,h=96,w=320):
        super().__init__('', ['synthetic 1 l'],h,w,[0,-1,1],4,is_train=True)
        self.K=np.array([[.58,0,.5,0],[0,1.92,.5,0],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.resolution_aug=mode
    def check_depth(self):return False
    def get_color(self,folder,frame_index,side,do_flip):
        y,x=np.mgrid[:180,:600]
        im=np.stack([(x+frame_index*3)%256,(y*2)%256,(x//3+y)%256],-1).astype(np.uint8)
        im=Image.fromarray(im)
        return im.transpose(getattr(Image,'Transpose',Image).FLIP_LEFT_RIGHT) if do_flip else im

def basic():
    k=fake_k(64,128)[None]
    yy,xx=torch.meshgrid(torch.arange(64),torch.arange(128),indexing='ij')
    z=(5+.1*(xx-k[0,0,2])/k[0,0,0]+.2*(yy-k[0,1,2])/k[0,1,1])[None,None]
    z.requires_grad_();q=z.detach().clone().requires_grad_();v=torch.ones_like(z)
    loss,_=br_cross_scale(z,q,k,k,v,v,'band')
    assert abs(float(loss)-.001)<1e-5
    shifted,_=br_cross_scale(z,q*1.2,k,k,v,v,'band')
    assert float(shifted)>.1
    shifted.backward();assert z.grad is not None and q.grad is not None
    a=torch.eye(4)[None];a[:,0,0]=a[:,1,1]=1.5;a[:,0,2]=.25-32;a[:,1,2]=.25-16
    kh=a@k
    zh=(5+.1*(xx-kh[0,0,2])/kh[0,0,0]+.2*(yy-kh[0,1,2])/kh[0,1,1])[None,None]
    plane,_=br_cross_scale(z.detach(),zh,k,kh,v,v,'band')
    assert float(plane)<.002,float(plane)
    m=torch.zeros_like(z);m[...,10:50,20:100]=1
    x=torch.rand_like(z);old=br_masked_mean(x,br_erode(m))
    x=x+(1-m)*10000
    assert torch.allclose(old,br_masked_mean(x,br_erode(m)))
    # Identical reference camera must recover the original convolution, including edges.
    torch.manual_seed(0);old=CDilated(8,8,3,d=3,groups=8)
    module=BRConditionalDepthwise(old,(.58*640,1.92*192))
    f=torch.rand(2,8,12,40);camera=fake_k(192,640)[None].repeat(2,1,1)
    reference=module(f,(camera,(192,640)))
    assert (reference-old(f)).abs().max()<1e-4
    camera[1,:2,:]*=1.5
    both=module(f,(camera,(192,640)))
    parts=torch.cat([module(f[i:i+1],(camera[i:i+1],(192,640))) for i in range(2)])
    assert torch.allclose(both,parts,atol=1e-5)
    both.square().mean().backward()
    assert module.conv.weight.grad is not None and module.gate[-1].weight.grad is not None
    # Optional modules must not change base/decoder random initialization.
    for conditional in [False,True]:
        torch.manual_seed(0)
        e=networks.LiteMono(model='lite-mono-tiny',height=192,width=640,conditional_cdc=conditional)
        d=networks.DepthDecoder(e.num_ch_enc,scales=[0,1,2])
        if not conditional:base_e=e.state_dict();base_d=d.state_dict()
        else:
            for name,value in base_e.items():assert torch.equal(value,e.state_dict()[name]),name
            for name,value in base_d.items():assert torch.equal(value,d.state_dict()[name]),name
    print('PASS: geometry, scale sensitivity, masks, CDC reference/batch/backward, common initialization')

def pipeline():
    from trainer import Trainer
    from options import LiteMonoOptions
    opt=LiteMonoOptions().parser.parse_args(['--no_cuda','--height','96','--width','320',
                                           '--batch_size','1','--num_workers','0'])
    # Bypass I/O-heavy constructor; exercise the real batch/loss implementation.
    t=Trainer.__new__(Trainer);t.opt=opt;t.device=torch.device('cpu');t.epoch=9
    t.ssim=SSIM();t.num_scales=3
    t.backproject_depth={0:BackprojectDepth(1,96,320)}
    t.project_3d={0:Project3D(1,96,320)}
    for conditional in [False,True]:
        e=networks.LiteMono(model='lite-mono-tiny',height=192,width=640,conditional_cdc=conditional)
        d=networks.DepthDecoder(e.num_ch_enc,scales=[0,1,2]);t.models={'encoder':e,'depth':d}
        shift=torch.nn.Parameter(torch.tensor(.01))
        def poses(inputs,features):
            result={}
            for fid in [-1,1]:
                mat=torch.eye(4)[None].clone();mat[:,0,3]=shift*fid
                result[('cam_T_cam',0,fid)]=mat
            return result
        t.predict_poses=poses
        for mode in ['none','ra','log','band']:
            opt.consistency=mode
            batch=next(iter(DataLoader(FakeDataset('pad3'),batch_size=1,num_workers=0)))
            out,loss=t.process_batch(batch)
            assert torch.isfinite(loss['loss']),mode
            loss['loss'].backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
            print('PASS pipeline',conditional,mode,float(loss['loss'].detach()),
                  {k:float(v.detach()) for k,v in loss.items() if 'valid' in k})
            e.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True)
    # Test frame/K alignment in the repeated-view control.
    item=FakeDataset('repeat3')[0]
    for name in ['L','H']:
        assert torch.equal(item[('color',0,0)],item[('br_color',name,0,0)])
        assert torch.allclose(item[('K',0)],item[('br_K',name,0)])
    print('PASS: repeat3 control')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    basic()
    if a.pipeline:pipeline()