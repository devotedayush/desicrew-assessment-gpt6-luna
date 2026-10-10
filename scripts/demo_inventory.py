"""Execute five questions against the clearly synthetic demo workbook."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from desicrew.inventory import InventoryAgent,load_workbook
from desicrew.search import PublicDefinitionSearch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-search',action='store_true',help='Permit public conceptual Wikipedia requests; never sends rows')
    args=parser.parse_args()
    workbook=ROOT/'data/synthetic/inventory.xlsx'
    context=load_workbook(workbook.read_bytes(),workbook.name)
    agent=InventoryAgent(context,search_client=PublicDefinitionSearch() if args.live_search else None)
    prompts=['How many products are there?','What is total stock quantity?','What is total inventory value?',
             'Which products are below reorder level?','What is inventory turnover?']
    exchanges=[{'question':q,'response':agent.ask(q)} for q in prompts]
    for turn in exchanges[:4]:
        if turn['response']['result'] is None:
            raise RuntimeError('Synthetic inventory query failed: '+turn['response']['answer'])
    output=ROOT/'artifacts/inventory_demo.json'
    output.write_text(json.dumps({'data':'synthetic only; not supplied assessment workbook','model':'none; deterministic planner',
                                 'search':'live public search' if args.live_search else 'disabled', 'exchanges':exchanges},indent=2)+'\n')
    print(json.dumps({'queries_executed':4,'context_questions':1,'mode':'synthetic-demo','output':str(output)},indent=2))
if __name__=='__main__':main()
