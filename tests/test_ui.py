"""Streamlit's supported AppTest validates UI callbacks without a browser.

These are UI execution checks, not screenshot evidence.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def _page(name):
    script=f'from desicrew.ui import sidebar_settings, {name}\nsidebar_settings()\n{name}()'
    return AppTest.from_string(script,default_timeout=30).run()

def _button(app,label):
    return next(button for button in app.button if button.label==label)

def test_support_replay_reset_and_repeat():
    app=_page('support_page')
    assert not app.exception
    _button(app,'Replay 12-turn demonstration').click().run()
    assert not app.exception
    assert len(app.session_state['support_history'])==12
    assert len(app.chat_message)==24
    for turn in app.session_state['support_history']:
        assert turn['response']['citations']
    _button(app,'Reset support session').click().run()
    assert not app.exception
    assert len(app.session_state['support_history'])==0
    app.chat_input[0].set_value('How do I invite teammates?').run()
    assert not app.exception
    assert len(app.session_state['support_history'])==1

def test_document_batch_and_repeated_extraction():
    app=_page('documents_page')
    assert not app.exception
    next(c for c in app.checkbox if c.label=='Use ten synthetic text fixtures instead').check().run()
    _button(app,'Classify and extract').click().run()
    assert not app.exception
    results=app.session_state['document_results']
    assert len(results)==10
    assert sum(len(r['fields']) for r in results)==44
    _button(app,'Classify and extract').click().run()
    assert not app.exception
    assert len(app.session_state['document_results'])==10

def test_inventory_query_and_clear():
    app=_page('inventory_page')
    assert not app.exception
    app.chat_input[0].set_value('How many products are there?').run()
    assert not app.exception
    history=app.session_state['inventory_history']
    assert len(history)==1
    assert history[0]['response']['mode']=='synthetic-demo'
    assert history[0]['response']['code']
    assert history[0]['response']['result'] is not None
    _button(app,'Clear inventory chat').click().run()
    assert not app.exception
    assert len(app.session_state['inventory_history'])==0

def test_unreadable_text_batch_helper(tmp_path):
    from desicrew.ui import _safe_document_result
    from desicrew.documents import DocumentProcessor
    file=tmp_path/'invalid.txt'
    file.write_bytes(b'\xff\xfe\x80')
    result=_safe_document_result(DocumentProcessor(),file,'pan_card')
    assert result['needs_review']
    assert len(result['fields'])==4
    assert all(field['value'] is None for field in result['fields'].values())
    assert result['ocr']['engine']=='unreadable_input'

def test_navigation_entrypoint_renders_default_page():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert any(title.value=='Ask your inventory' for title in app.title)
