from pathlib import Path
import zipfile
from scripts.package_source import create_package

def test_private_data_env_and_outputs_excluded(tmp_path):
    root=tmp_path/'source';root.mkdir()
    for name in ['app.py','README.md','.env','.env.example','data/private/id.png','data/synthetic/demo.txt','outputs/private_samples/extraction.json','artifacts/private/secret.json']:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic test contents')
    zip_path=tmp_path/'source.zip'
    result=create_package(zip_path,root)
    assert result['file_count']==5
    with zipfile.ZipFile(zip_path) as archive:
        names=archive.namelist()
    assert 'desicrew-assessment/.env.example' in names
    assert all('/private/' not in n and '/outputs/' not in n for n in names)
    assert 'desicrew-assessment/.env' not in names


def test_symlink_tree_rejected(tmp_path):
    import pytest
    root=tmp_path/'source';root.mkdir()
    outside=tmp_path/'private';outside.mkdir()
    (outside/'identity.txt').write_text('synthetic secret')
    (root/'docs').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        create_package(tmp_path/'source.zip',root)


def test_secret_suffix_rejected(tmp_path):
    import pytest
    root=tmp_path/'source';(root/'docs').mkdir(parents=True)
    (root/'docs/example.key').write_text('synthetic key')
    with pytest.raises(ValueError,match='Credential'):
        create_package(tmp_path/'source.zip',root)
