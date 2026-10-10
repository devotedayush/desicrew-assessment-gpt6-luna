"""Package reproducible source and synthetic evidence, never supplied private files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
TOP_FILES={'app.py','README.md','BRIEF.md','requirements.txt','requirements-dev.txt','requirements-lock.txt','requirements-handwriting.txt','requirements-got-ocr.txt','requirements-inventory-model.txt','requirements-macos-vision.txt',
           'pyproject.toml','Dockerfile','docker-compose.yml','.gitignore','.dockerignore','.env.example'}
TREES=('desicrew','tests','docs','scripts','data/synthetic','.streamlit')
SYNTHETIC_ARTIFACTS=('artifacts/support_demo.json','artifacts/inventory_demo.json',
                     'artifacts/documents_synthetic','artifacts/documents_printed_ocr',
                     'artifacts/support_evaluation.json','artifacts/live_definition_search.json',
                     'artifacts/q1_model_evaluation_summary.json','artifacts/inventory_model_final_qa_public.json','artifacts/q3_evaluation_summary.json',
                     'artifacts/verification',
                     'artifacts/ui_evidence_support_documents.json',
                     'artifacts/ui_evidence_inventory.json','artifacts/ui_evidence_inventory_llm.json',
                     'artifacts/ui_inventory_transcript.json','artifacts/ui_inventory_transcript_llm.json','artifacts/ui_inventory_definition_transcript.json','artifacts/ui_documents_image_transcript.json',
                     'artifacts/screenshots/q1_inventory_execution.png',
                     'artifacts/screenshots/q1_definition_search.png',
                     'artifacts/screenshots/q1_llm_execution.png',
                     'artifacts/screenshots/q1_llm_failure.png',
                     'artifacts/screenshots/q1_invalid_workbook.png',
                     'artifacts/screenshots/q1_invalid_workbook_llm.png',
                     'artifacts/screenshots/q2_support_session.png',
                     'artifacts/screenshots/q2_support_middle.png',
                     'artifacts/screenshots/q2_support_end.png',
                     'artifacts/screenshots/q2_out_of_scope.png',
                     'artifacts/screenshots/q3_synthetic_review.png',
                     'artifacts/screenshots/q3_synthetic_fields.png',
                     'artifacts/screenshots/q3_synthetic_image_ocr.png',
                     'artifacts/screenshots/q3_unreadable_input.png')

SECRET_SUFFIXES={".pem",".key",".token",".p12",".pfx",".jks",".keystore"}

def selected_files(root=ROOT):
    root=Path(root)
    if root.is_symlink():
        raise ValueError("Source root cannot be a symlink")
    paths=[]
    for name in TOP_FILES:
        path=root/name
        if path.is_file():paths.append(path)
    for name in (*TREES,*SYNTHETIC_ARTIFACTS):
        path=root/name
        if path.is_symlink():
            raise ValueError("Selected source tree cannot be a symlink")
        paths.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [])
    files=[]
    for path in paths:
        rel=path.relative_to(root)
        ancestors=[root.joinpath(*rel.parts[:i]) for i in range(1,len(rel.parts)+1)]
        if any(ancestor.is_symlink() for ancestor in ancestors):
            raise ValueError('Symlinks cannot enter source package')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source path escapes package root')
        if path.suffix.lower() in SECRET_SUFFIXES:
            raise ValueError('Credential/private-key files cannot enter source package')
        if '__pycache__' in rel.parts or path.suffix in {'.pyc','.log'}:
            continue
        if any(part in {'.venv','.qa-venv','.hwr-venv','.got-venv','private','private_samples','.git'} for part in rel.parts):
            raise ValueError('Private path cannot enter source package')
        if path.name.startswith('.env') and path.name!='.env.example':
            raise ValueError('Environment secrets cannot enter source package')
        files.append(path)
    return sorted(set(files))

def create_package(output,root=ROOT):
    root=Path(root);output=Path(output)
    files=selected_files(root)
    entries=[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]
    manifest={'purpose':'Source and clearly synthetic evidence only; no supplied identity/bank/workbook files',
              'required_screenshot_evidence':'Genuine captures included' if any('/screenshots/' in e['path'] for e in entries) else 'No screenshots in this source root',
              'files':entries}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,arcname='desicrew-assessment/'+path.relative_to(root).as_posix())
        archive.writestr('desicrew-assessment/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():raise RuntimeError('Package ZIP integrity check failed')
        for name in archive.namelist():
            if any(part in {'private','private_samples','.venv','.qa-venv','.hwr-venv','.got-venv','.git','__pycache__'} for part in Path(name).parts):
                raise RuntimeError('A forbidden private path entered the archive')
    return {'file_count':len(entries)+1,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'path':str(output)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'desicrew-assessment-source.zip')
    args=parser.parse_args()
    print(json.dumps(create_package(args.output),indent=2))
if __name__=='__main__':main()
