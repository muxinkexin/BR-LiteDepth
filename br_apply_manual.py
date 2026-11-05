"""Temporary patch installer. Default is a dry run; --apply writes local files."""
import argparse,ast,json,re,shutil
from pathlib import Path

ALLOWED={'options.py','train.py','trainer.py','layers.py',
         'datasets/mono_dataset.py','networks/depth_encoder.py'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manual',required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    text=Path(a.manual).read_text()
    blocks=re.findall(r'^<!-- BR_EDIT_START -->\n(.*?)^<!-- BR_EDIT_END -->$',text,re.M|re.S)
    assert len(blocks)==24,f'Expected 24 edits, found {len(blocks)}'
    originals={};changed={}
    for number,block in enumerate(blocks,1):
        meta=re.search(r'^<!-- BR_EDIT_META (.*?) -->$',block,re.M)
        assert meta,number
        data=json.loads(meta.group(1));file=data['file'];assert file in ALLOWED,file
        path=Path(file);assert path.is_file(),f'Run inside repository root: {file}'
        if file not in originals:
            assert not Path(file+'.br_v3.bak').exists(),f'Backup exists; do not apply twice: {file}'
            originals[file]=path.read_text();changed[file]=originals[file]
        snippets=re.findall(r'```python\n(.*?)\n```',block,re.S)
        if data['kind']=='append':
            assert len(snippets)==1
            changed[file]=changed[file].rstrip()+'\n\n'+snippets[0].rstrip()+'\n'
        else:
            assert len(snippets)==2
            old,new=snippets
            assert changed[file].count(old)==1,f'Edit {number} does not match uniquely: {file}'
            changed[file]=changed[file].replace(old,new,1)
    assert set(changed)==ALLOWED
    for file,code in changed.items():
        ast.parse(code,filename=file)
        print(file,'OK',len(code.splitlines()),'lines')
    if not a.apply:
        print('Dry run passed. Review the manual and add --apply after E0 finishes.');return
    for file in sorted(changed):
        path=Path(file);shutil.copy2(path,Path(file+'.br_v3.bak'))
        path.write_text(changed[file])
    print('Applied 24 edits to 6 existing files; no new source directory created.')

if __name__=='__main__':main()