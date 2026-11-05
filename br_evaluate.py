"""Temporary root-level evaluator for the BR-LiteDepth manual."""
import os,sys,tempfile
if __name__ == '__main__':
    if 'LD_LIBRARY_PATH' in os.environ or 'LD_PRELOAD' in os.environ:
        env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
        os.execve(sys.executable,[sys.executable]+sys.argv,env)
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'litemono-matplotlib'))
import argparse,csv,hashlib,json,time,inspect
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import networks
from layers import disp_to_depth

SIZES=['640x192','416x128','512x160','832x256','1024x320']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metric(g,z):
    q=np.maximum(g/z,z/g)
    return [np.mean(abs(g-z)/g),np.mean((g-z)**2/g),np.sqrt(np.mean((g-z)**2)),
            np.sqrt(np.mean((np.log(g)-np.log(z))**2)),
            np.mean(q<1.25),np.mean(q<1.25**2),np.mean(q<1.25**3)]

def report(out,sizes,gt_path):
    gt=np.load(gt_path,allow_pickle=True)['data']
    predictions={s:np.load(out/s/'disps_eigen_split.npy',mmap_mode='r') for s in sizes}
    count=len((out/'640x192'/'files.txt').read_text().splitlines())
    assert len(gt)==count and all(len(v)==count for v in predictions.values())
    columns=['abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3']
    rows=[]
    for i,g0 in enumerate(gt):
        h,w=g0.shape; mask=(g0>1e-3)&(g0<80)
        c=np.array([.40810811*h,.99189189*h,.03594771*w,.96405229*w]).astype(int)
        crop=np.zeros_like(mask);crop[c[0]:c[1],c[2]:c[3]]=True;mask &= crop
        if not mask.any():raise ValueError(f'GT image {i} has no valid crop pixels')
        g=g0[mask].astype(np.float64);zs={}
        for s,v in predictions.items():
            inv=cv2.resize(np.asarray(v[i]),(w,h),interpolation=cv2.INTER_LINEAR)
            assert np.isfinite(inv).all() and (inv>0).all()
            zs[s]=(1.0/inv[mask]).astype(np.float64)
        zr=zs['640x192'];alpha=np.median(g)/np.median(zr)
        for s,z in zs.items():
            own=np.median(g)/np.median(z)
            row=dict(image_index=i,resolution=s,
                     drift=float(abs(np.median(np.log(z)-np.log(zr)))),reference_alpha=float(alpha))
            for tag,value in [('standard',own),('shared',alpha)]:
                row.update({tag+'_'+k:float(v) for k,v in zip(columns,metric(g,np.clip(z*value,1e-3,80)))})
            rows.append(row)
    with (out/'per_image.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    fields=['drift']+['standard_'+k for k in columns]+['shared_'+k for k in columns]
    summary={s:{k:float(np.mean([r[k] for r in rows if r['resolution']==s])) for k in fields} for s in sizes}
    others=[s for s in sizes if s!='640x192']
    if others:summary['nonreference_mean']={k:float(np.mean([summary[s][k] for s in others])) for k in fields}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--data_path')
    p.add_argument('--ext',choices=['png','jpg'],default='png')
    p.add_argument('--split_file',default='splits/eigen/test_files.txt')
    p.add_argument('--gt',default='splits/eigen/gt_depths.npz')
    p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--sizes',nargs='+',default=SIZES)
    p.add_argument('--reuse_predictions',action='store_true')
    p.add_argument('--benchmark_only',action='store_true')
    p.add_argument('--no_gt',action='store_true')
    p.add_argument('--torch_threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.torch_threads);cv2.setNumThreads(0)
    device=torch.device(a.device)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    wd=Path(a.weights);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    enc=torch.load(wd/'encoder.pth',map_location='cpu',weights_only=True)
    dec=torch.load(wd/'depth.pth',map_location='cpu',weights_only=True)
    op=wd.parent/'opt.json';opt=json.loads(op.read_text()) if op.is_file() else {}
    model_name=enc.get('model_name',opt.get('model','lite-mono-tiny'))
    conditional=bool(enc.get('conditional_cdc',opt.get('conditional_cdc',False)))
    dh,dw=int(enc['height']),int(enc['width'])
    lo=float(enc.get('min_depth',opt.get('min_depth',0.1)))
    hi=float(enc.get('max_depth',opt.get('max_depth',100.0)))
    scales=opt.get('scales',[0,1,2]);assert list(scales)==[0,1,2]
    encoder=networks.LiteMono(model=model_name,height=dh,width=dw,conditional_cdc=conditional).to(device).eval()
    decoder=networks.DepthDecoder(encoder.num_ch_enc,scales=scales).to(device).eval()
    known={'height','width','use_stereo','model_name','conditional_cdc','min_depth','max_depth'}
    encoder.load_state_dict({k:v for k,v in enc.items() if k not in known},strict=True)
    decoder.load_state_dict(dec,strict=True)
    param_count=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in decoder.parameters())
    camera_supported = 'camera' in inspect.signature(encoder.forward).parameters
    def predict(x,k):
        return decoder(encoder(x,camera=k) if camera_supported else encoder(x))
    sizes=list(dict.fromkeys(a.sizes))
    if not a.benchmark_only and '640x192' not in sizes:raise ValueError('Need reference 640x192 for scale stability')
    rows=None
    if not a.benchmark_only:
        if not a.data_path:raise ValueError('--data_path required')
        rows=Path(a.split_file).read_text().splitlines();assert rows
    benchmark={}
    for size in sizes:
        w,h=map(int,size.split('x'));assert min(h,w)>0 and h%32==0 and w%32==0
        k=torch.tensor([[.58*w,0,.5*w,0],[0,1.92*h,.5*h,0],[0,0,1,0],[0,0,0,1]],device=device)[None]
        target=out/size;target.mkdir(exist_ok=True)
        meta=dict(encoder_sha256=sha(wd/'encoder.pth'),decoder_sha256=sha(wd/'depth.pth'),
                  train_hw=[dh,dw],eval_hw=[h,w],model=model_name,conditional_cdc=conditional,
                  min_depth=lo,max_depth=hi,image_ext=a.ext,params=param_count,
                  split_sha256=sha(a.split_file) if rows is not None else None,post_process=False)
        dest=target/'disps_eigen_split.npy';meta_path=target/'metadata.json'
        if a.benchmark_only:
            x=torch.rand(1,3,h,w,device=device)
            with torch.inference_mode():
                for _ in range(50):predict(x,k)
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(200):
                    if device.type=='cuda':
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();predict(x,k);end.record();end.synchronize()
                        times.append(start.elapsed_time(end))
                    else:
                        t=time.perf_counter();predict(x,k);times.append((time.perf_counter()-t)*1000)
            benchmark[size]=dict(median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
                                 params=param_count,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            continue
        if a.reuse_predictions:
            assert dest.is_file() and meta_path.is_file()
            assert json.loads(meta_path.read_text())==meta,'Cached prediction configuration differs'
            assert (target/'files.txt').read_text()=='\n'.join(rows)+'\n'
            assert np.load(dest,mmap_mode='r').shape==(len(rows),h,w)
            continue
        if dest.exists():raise FileExistsError(f'{dest}: use a new output folder or --reuse_predictions')
        memory=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(rows),h,w))
        with torch.inference_mode():
            for i,row in enumerate(rows):
                folder,frame,side=row.split();side={'l':2,'r':3,'2':2,'3':3}[side]
                path=Path(a.data_path)/folder/f'image_0{side}'/'data'/f'{int(frame):010d}.{a.ext}'
                with Image.open(path) as im:
                    im=im.convert('RGB').resize((w,h),resample=getattr(Image,'Resampling',Image).LANCZOS)
                    arr=np.array(im,dtype=np.float32)/255.0
                x=torch.from_numpy(arr).permute(2,0,1)[None].to(device)
                raw=predict(x,k)[('disp',0)]
                assert tuple(raw.shape)==(1,1,h,w)
                inverse,_=disp_to_depth(raw,lo,hi)
                memory[i]=inverse[0,0].cpu().numpy()
                if i==0:print('actual tensor',tuple(x.shape),'train constructor',(dh,dw),'conditional',conditional)
        memory.flush();del memory
        # Commit metadata last: an interrupted file is not a valid reusable result.
        (target/'files.txt').write_text('\n'.join(rows)+'\n')
        meta_path.write_text(json.dumps(meta,indent=2))
    if a.benchmark_only:
        (out/'benchmark.json').write_text(json.dumps(benchmark,indent=2));print(json.dumps(benchmark,indent=2))
    elif not a.no_gt:report(out,sizes,a.gt)

if __name__=='__main__':main()