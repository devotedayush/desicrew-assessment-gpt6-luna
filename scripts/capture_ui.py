"""Browser smoke checks and synthetic-only screenshots of the real Streamlit UI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8501')
    parser.add_argument('--chromium',default='/usr/bin/chromium')
    args=parser.parse_args()
    out=ROOT/'artifacts/screenshots';out.mkdir(parents=True,exist_ok=True)
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
        page.set_default_timeout(20000)
        page.goto(args.base_url+'/support')
        page.get_by_role('heading',name='Support that remembers').wait_for()
        page.get_by_role('button',name='Replay 12-turn demonstration').click()
        page.get_by_text('12 full exchanges this session',exact=False).wait_for()
        page.get_by_text('How do I invite teammates?',exact=True).first.wait_for()
        page.screenshot(path=str(out/'q2_support_session.png'),full_page=True)
        checks.append({'check':'Q2 real 12-exchange replay','result':'passed'})
        page.get_by_role('button',name='Reset support session').click()
        page.get_by_text('0 full exchanges this session',exact=False).wait_for()
        checks.append({'check':'Q2 reset session','result':'passed'})
        page.goto(args.base_url+'/documents')
        page.get_by_role('heading',name='Review document extraction').wait_for()
        page.get_by_label('Use ten synthetic text fixtures instead').check()
        page.get_by_role('button',name='Classify and extract').click()
        page.get_by_text('Fields represented',exact=True).wait_for()
        page.get_by_text('44',exact=True).wait_for()
        page.screenshot(path=str(out/'q3_synthetic_review.png'),full_page=True)
        checks.append({'check':'Q3 ten documents and44 represented fields','result':'passed'})
        page.get_by_role('button',name='Classify and extract').click()
        page.get_by_text('44',exact=True).wait_for()
        checks.append({'check':'Q3 repeated extraction stable','result':'passed'})
        page.goto(args.base_url)
        page.get_by_role('heading',name='Ask your inventory').wait_for()
        chat=page.get_by_placeholder('Ask about the selected inventory workbook')
        chat.fill('How many products are there?')
        chat.press('Enter')
        page.get_by_text('Execution mode: synthetic-demo',exact=True).wait_for()
        page.get_by_text('Generated code and executed result',exact=True).wait_for()
        page.screenshot(path=str(out/'q1_inventory_execution.png'),full_page=True)
        if page.get_by_text('The query could not be completed safely.',exact=True).count():
            raise AssertionError('Inventory execution failed')
        checks.append({'check':'Q1 synthetic query generated and executed','result':'passed'})
        page.get_by_role('button',name='Clear inventory chat').click()
        page.get_by_placeholder('Ask about the selected inventory workbook').wait_for()
        assert page.get_by_text('Execution mode: synthetic-demo',exact=True).count()==0
        checks.append({'check':'Q1 clear chat','result':'passed'})
        errors=page.locator('[data-testid="stException"]')
        if errors.count():
            raise AssertionError(errors.all_text_contents())
        checks.append({'check':'page navigation and no Streamlit exception','result':'passed'})
        browser.close()
    (ROOT/'artifacts/ui_checks.json').write_text(json.dumps({'data':'synthetic only','checks':checks},indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
