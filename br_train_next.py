"""Temporary launcher: inherit the actual E0 training options, no new config tree."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path
from options import LiteMonoOptions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference_opt',required=True)
    p.add_argument('--stage',choices=['E0R','EC','E1','E2','E2log','E3','E4','E5'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--name')
    p.add_argument('--lambda_cs',type=float)
    p.add_argument('--lambda_band',type=float)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    path=Path(a.reference_opt);original=json.loads(path.read_text())
    assert original.get('model')=='lite-mono-tiny'
    assert (original['height'],original['width'])==(192,640)
    assert original.get('mypretrain'),'E0 must use a common ImageNet pretrain file'
    assert not original.get('load_weights_folder'),'Fine-tuning E0 needs a separate matched protocol'
    assert Path(original['mypretrain']).expanduser().is_file(),'Pretrain path is not available in this working copy'
    modes={'E0R':('none','none',False),'EC':('repeat3','none',False),
           'E1':('pad3','none',False),'E2':('pad3','ra',False),
           'E2log':('pad3','log',False),'E3':('pad3','band',False),
           'E4':('pad3','ra',True),'E5':('pad3','band',True)}
    aug,consistency,conditional=modes[a.stage]
    opts=original.copy()
    opts.update(seed=a.seed,resolution_aug=aug,consistency=consistency,conditional_cdc=conditional,
                lambda_cs=1.0,lambda_band=0.2,consistency_start_epoch=5,consistency_ramp_epochs=5,
                model_name=a.name or (f'{a.stage}_pretrained_s{a.seed}'+('_smoke' if a.smoke else '')),
                load_weights_folder=None)
    if a.smoke:opts['num_epochs']=1
    if a.lambda_cs is not None:opts['lambda_cs']=a.lambda_cs
    if a.lambda_band is not None:opts['lambda_band']=a.lambda_band
    parser=LiteMonoOptions().parser
    command=[sys.executable,'train.py']
    for action in parser._actions:
        if action.dest in ['help','load_weights_folder','models_to_load'] or action.dest not in opts:continue
        value=opts[action.dest]
        if value is None:continue
        flag=next((s for s in action.option_strings if s.startswith('--')),None)
        if flag is None:continue
        if isinstance(action,argparse._StoreTrueAction):
            if value:command.append(flag)
        elif isinstance(action,argparse._StoreFalseAction):
            if not value:command.append(flag)
        elif isinstance(value,list):command += [flag]+[str(x) for x in value]
        else:command += [flag,str(value)]
    parser.parse_args(command[2:]) # catch unsupported flags before starting a job
    print(shlex.join(command),flush=True)
    if not a.run:
        print('Dry run only. Review the command, then add --run.');return
    run_dir=Path(opts['log_dir']).expanduser()/opts['model_name']
    if run_dir.exists():raise FileExistsError(f'Use a new model_name: {run_dir}')
    run_dir.mkdir(parents=True)
    weight_hash=hashlib.sha256()
    with Path(opts['mypretrain']).expanduser().open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):weight_hash.update(block)
    record=dict(pretrain_sha256=weight_hash.hexdigest(),reference_opt=str(path.resolve()),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                command=command,stage=a.stage,smoke=a.smoke)
    (run_dir/'br_launch.json').write_text(json.dumps(record,indent=2))
    with (run_dir/'console.log').open('w',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line)
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate();process.wait();raise
    if code:raise SystemExit(code)

if __name__=='__main__':main()