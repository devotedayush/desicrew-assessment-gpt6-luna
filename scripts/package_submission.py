"""Build a LOCAL reviewer submission, with required sensitive Q3 JSON only.

Public-safe source ZIP is separate. This combined ZIP must stay private.
No source scans, independent visual answer references, model weights, credentials,
virtual environments, or private crop/model logs are copied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from package_source import create_package

ROOT=Path(__file__).resolve().parents[1]

def build(output:Path, reviewer_dir:Path, root:Path=ROOT):
    output=Path(output); reviewer_dir=Path(reviewer_dir).resolve(); root=Path(root)
    results=json.loads((reviewer_dir/'results.json').read_text())
    report=json.loads((reviewer_dir/'review_report.json').read_text())
    if len(results)!=10 or sum(len(r['fields']) for r in results)!=44:
        raise ValueError('Expected the ten required documents and 44 fields')
    # Match per-document JSON against the batch, so no unrelated private JSON enters.
    docs=[]
    for p in sorted(reviewer_dir.glob('*.json')):
        if p.name in {'results.json','review_report.json','run_summary.json'}: continue
        r=json.loads(p.read_text())
        if r in results: docs.append((p,r))
    if len(docs)!=10: raise ValueError('Expected ten matching per-document JSON files')
    if not isinstance(report,dict): raise ValueError('Invalid review report')
    files=[]
    with tempfile.TemporaryDirectory(prefix='desicrew-package-') as tmp:
        source=Path(tmp)/'source.zip'; create_package(source,root)
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            with zipfile.ZipFile(source) as z:
                for name in z.namelist(): archive.writestr(name,z.read(name))
            for p,r in docs:
                name='desicrew-assessment/reviewer_outputs/'+p.name
                raw=p.read_bytes();archive.writestr(name,raw)
                files.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
            for name in ('results.json','review_report.json'):
                raw=(reviewer_dir/name).read_bytes()
                target='desicrew-assessment/reviewer_outputs/'+name
                archive.writestr(target,raw)
                files.append({'path':target,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
            readme=('PRIVATE REVIEWER OUTPUTS\n\nThese are actual automatic extractions of the supplied scans, not an answer key. '
                    'They contain identity/bank information and must not be published. All 44 fields require review. '
                    'See docs/Q3_LOCAL_EVALUATION.md for the backend, diagnostic results and limitations. '
                    'Original scans, private references and model weights are excluded.\n')
            archive.writestr('desicrew-assessment/reviewer_outputs/README.txt',readme)
            archive.writestr('desicrew-assessment/SUBMISSION_MANIFEST.json',json.dumps({
                'purpose':'LOCAL private reviewer submission; never publish this archive',
                'required_documents':10,'field_targets':44,'reviewer_json_files':files,
                'source_manifest':'PACKAGE_MANIFEST.json'},indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip():raise RuntimeError('ZIP integrity failed')
    return {'path':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'required_document_jsons':len(docs),'field_targets':44}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewer-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'DesiCrew-Submission-LOCAL.zip')
    args=parser.parse_args();print(json.dumps(build(args.output,args.reviewer_dir),indent=2))
if __name__=='__main__':main()
