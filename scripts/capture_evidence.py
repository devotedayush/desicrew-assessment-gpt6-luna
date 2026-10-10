"""Capture genuine UI evidence, using only clearly synthetic public fixtures.

Run against the Linux UI for inventory sandbox execution. Chromium's OS sandbox
is enabled by default; this script never changes the application's code sandbox.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8501')
    parser.add_argument('--chromium', help='Optional installed Chromium/Chrome executable')
    parser.add_argument('--pages', nargs='+', choices=['inventory','support','documents'], default=['inventory','support','documents'])
    parser.add_argument('--llm', action='store_true', help='Enable already-configured model for synthetic inventory evidence')
    args = parser.parse_args()
    out = ROOT/'artifacts/screenshots'; out.mkdir(parents=True, exist_ok=True)
    checks = []; captures = []
    def capture(page, name, description):
        target = out/name
        print('Capturing', name, flush=True)
        page.wait_for_timeout(250)  # Let native accordion/scroll animation settle.
        page.screenshot(path=str(target), full_page=True)
        captures.append({'path': 'artifacts/screenshots/'+name, 'description': description,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'captured_utc': datetime.now(timezone.utc).isoformat()})
    with sync_playwright() as p:
        opts = {'headless': True, 'chromium_sandbox': True}
        if args.chromium: opts['executable_path'] = args.chromium
        browser = p.chromium.launch(**opts)
        page = browser.new_page(viewport={'width':1440,'height':1080}, device_scale_factor=1)
        page.set_default_timeout(90000)
        if 'support' in args.pages:
            page.goto(args.base_url+'/support')
            page.get_by_role('heading', name='Support that remembers').wait_for()
            page.get_by_role('button', name='Replay 12-turn demonstration').click()
            page.get_by_text('12 full exchanges this session', exact=False).wait_for()
            page.locator('[data-testid=stAppScrollToBottomContainer]').evaluate('(el) => { el.scrollTop = 0; }')
            page.get_by_text('Evidence and session context', exact=True).first.click()
            capture(page,'q2_support_session.png','Real UI replay: first exchanges of 12 sequential cited turns')
            page.get_by_text('Can renewal charges be refunded?', exact=True).scroll_into_view_if_needed()
            capture(page,'q2_support_middle.png','Real UI replay: middle exchanges and topic switches')
            page.get_by_text('What about the other link?', exact=True).scroll_into_view_if_needed()
            capture(page,'q2_support_end.png','Real UI replay: final exchanges of 12 sequential turns')
            checks.append({'check':'Q2 12-exchange replay', 'result':'passed'})
            page.get_by_role('button', name='Reset support session').click()
            page.get_by_text('0 full exchanges this session', exact=False).wait_for()
            chat=page.get_by_placeholder('Ask about MeadowDesk policies')
            chat.fill('What is the weather in Delhi?'); chat.press('Enter')
            page.get_by_text('Evidence and session context', exact=True).wait_for()
            capture(page,'q2_out_of_scope.png','Unknown topic handled by cited scope response')
            checks.append({'check':'Q2 reset and out-of-scope response', 'result':'passed'})
        if 'documents' in args.pages:
            page.goto(args.base_url+'/documents')
            page.get_by_role('heading', name='Review document extraction').wait_for()
            page.get_by_text('Use ten synthetic text fixtures instead', exact=True).click()
            page.get_by_role('button', name='Classify and extract', exact=True).click()
            page.get_by_text('Fields represented', exact=True).wait_for()
            page.get_by_text('44', exact=True).wait_for()
            page.get_by_text('Fields represented', exact=True).scroll_into_view_if_needed()
            capture(page,'q3_synthetic_review.png','Ten synthetic text documents, all 44 field targets and review statuses')
            page.get_by_text('Structured extraction and provenance', exact=True).first.scroll_into_view_if_needed()
            page.get_by_text('Structured extraction and provenance', exact=True).first.click()
            capture(page,'q3_synthetic_fields.png','Required fields, confidence and provenance for synthetic source')
            checks.append({'check':'Q3 synthetic schema coverage', 'result':'passed'})
            page.close()
            page = browser.new_page(viewport={'width':1440,'height':1080}, device_scale_factor=1)
            page.goto(args.base_url+'/documents')
            page.get_by_role('heading', name='Review document extraction').wait_for()
            upload=page.locator('input[type=file]')
            upload.set_input_files([str(ROOT/'data/synthetic/documents/nach_ecs_mandate.png')])
            page.get_by_text('nach_ecs_mandate.png', exact=True).wait_for()
            page.get_by_role('button', name='Classify and extract', exact=True).click()
            page.get_by_text('Download extraction JSON', exact=True).wait_for()
            with page.expect_download() as image_download:
                page.get_by_role('button', name='Download extraction JSON').click()
            image_results = json.loads(Path(image_download.value.path()).read_text())
            page.wait_for_timeout(750)  # Download widgets can rerender the real page.
            assert len(image_results) == 1 and image_results[0]['source_file'] == 'nach_ecs_mandate.png'
            assert image_results[0]['document_type'] == 'nach_ecs_mandate' and len(image_results[0]['fields']) == 5
            assert image_results[0]['ocr']['is_image_ocr'] and image_results[0]['ocr']['pages'] > 0
            (ROOT/'artifacts/ui_documents_image_transcript.json').write_text(json.dumps(image_results,indent=2)+'\n')
            page.get_by_text('Structured extraction and provenance', exact=True).first.scroll_into_view_if_needed()
            capture(page,'q3_synthetic_image_ocr.png','Actual local OCR of synthetic bank mandate image; no supplied private scan')
            checks.append({'check':'Q3 image OCR in UI', 'result':'passed'})
            page.close()
            page = browser.new_page(viewport={'width':1440,'height':1080}, device_scale_factor=1)
            page.goto(args.base_url+'/documents')
            page.get_by_role('heading', name='Review document extraction').wait_for()
            upload = page.locator('input[type=file]')
            upload.set_input_files({'name':'unreadable-demo.png', 'mimeType':'image/png','buffer':b'not an image'})
            page.get_by_text('unreadable-demo.png', exact=True).wait_for()
            page.get_by_role('button', name='Classify and extract', exact=True).click()
            page.get_by_text('file_read_or_ocr_failed', exact=True).wait_for()
            page.get_by_text('file_read_or_ocr_failed', exact=True).scroll_into_view_if_needed()
            capture(page,'q3_unreadable_input.png','Genuine unreadable-image failure routed to review without stopping the app')
            checks.append({'check':'Q3 unreadable input review', 'result':'passed'})
        if 'inventory' in args.pages:
            page.goto(args.base_url)
            page.get_by_role('heading', name='Ask your inventory').wait_for()
            if args.llm: page.get_by_text('Enable configured LLM', exact=True).click()
            chat=page.get_by_placeholder('Ask about the selected inventory workbook')
            chat.fill('How many rows are in worksheet Inventory?'); chat.press('Enter')
            page.get_by_text('Execution mode: '+('llm' if args.llm else 'synthetic-demo'),exact=True).wait_for()
            with page.expect_download() as download_info:
                page.get_by_role('button', name='Download this inventory transcript').click()
            transcript = json.loads(Path(download_info.value.path()).read_text())
            page.wait_for_timeout(750)
            response = transcript[-1]['response']
            if response.get('result') is None:
                capture(page,'q1_llm_failure.png','Actual generated query failed; no success claim')
                raise AssertionError('Inventory result was null: '+response.get('answer',''))
            import sys
            if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
            from desicrew.inventory import load_workbook
            source = ROOT/'data/synthetic/inventory.xlsx'
            workbook = load_workbook(source.read_bytes(), source.name)
            expected = len(next(iter(workbook.sheets.values())))
            assert response['result'].get('type') == 'scalar' and response['result'].get('value') == expected, 'Executed row count disagrees with independent workbook count'
            (ROOT/'artifacts'/('ui_inventory_transcript'+('_llm' if args.llm else '')+'.json')).write_text(json.dumps(transcript,indent=2)+'\n')
            capture(page,'q1_'+('llm' if args.llm else 'inventory')+'_execution.png','Synthetic workbook query: genuine '+('local LLM' if args.llm else 'deterministic planner')+' code executed in Linux sandbox')
            checks.append({'check':'Q1 synthetic query execution '+('LLM' if args.llm else 'offline'), 'result':'passed'})
            if not args.llm:
                chat.fill('Define reorder point'); chat.press('Enter')
                page.get_by_text('Public definition search sources', exact=True).wait_for()
                page.get_by_text('Public definition search sources', exact=True).click()
                with page.expect_download() as definition_download:
                    page.get_by_role('button', name='Download this inventory transcript').click()
                definition_transcript = json.loads(Path(definition_download.value.path()).read_text())
                definition_response = definition_transcript[-1]['response']
                assert definition_response['result']['type'] == 'context' and definition_response['search_sources']
                assert definition_response['summary_mode'] == 'built-in-context'
                (ROOT/'artifacts/ui_inventory_definition_transcript.json').write_text(json.dumps(definition_transcript,indent=2)+'\n')
                page.wait_for_timeout(750)
                page.get_by_text('Public definition search sources', exact=True).scroll_into_view_if_needed()
                capture(page,'q1_definition_search.png','Real public definition lookup with retrieved source links; built-in wording is explicitly disclosed')
                checks.append({'check':'Q1 real public search in UI, distinct from built-in wording', 'result':'passed'})
            page.locator('input[type=file]').set_input_files({'name':'invalid-demo.xlsx', 'mimeType':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'buffer':b'not a workbook'})
            page.get_by_text('Could not load workbook:', exact=False).wait_for()
            capture(page,'q1_invalid_workbook'+('_llm' if args.llm else '')+'.png','Genuine invalid workbook rejected with visible load error')
            checks.append({'check':'Q1 invalid workbook handling', 'result':'passed'})
        assert not page.locator('[data-testid="stException"]').count()
        browser.close()
    report={'data':'Public evidence uses fictional KB and synthetic fixtures only', 'base_url': args.base_url,
            'pages':args.pages, 'checks':checks, 'captures':captures}
    path=ROOT/'artifacts'/('ui_evidence_'+'_'.join(args.pages)+('_llm' if args.llm else '')+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__ == '__main__': main()
