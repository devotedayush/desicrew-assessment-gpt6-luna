"""Mocked HTTP tests; no credentials or live API calls."""
import json
import unittest
import urllib.error
from unittest.mock import patch
from desicrew.llm import ModelError, OpenAICompatibleClient
from desicrew.openai_luna import OpenAILunaClient
from desicrew.support import SupportAssistant
from pathlib import Path


class LunaTests(unittest.TestCase):
    def response(self, opener, content, status='completed'):
        opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps({
            'status': status, 'output': [
                {'type': 'reasoning', 'summary': []},
                {'type': 'message', 'content': content},
            ],
        }).encode()

    @patch('urllib.request.build_opener')
    def test_responses_transport_preserves_existing_json_contract(self, opener):
        self.response(opener, [{'type': 'output_text', 'text': '{"code":"result = len(df)"}'}])
        client = OpenAILunaClient('SYNTHETIC_TEST_TOKEN')
        result = client.complete([{'role': 'user', 'content': 'Return JSON code'}], True)
        self.assertEqual(json.loads(result)['code'], 'result = len(df)')
        request = opener.return_value.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/responses')
        self.assertEqual(payload['model'], 'gpt-6-luna')
        self.assertEqual(payload['text']['format']['type'], 'json_object')
        self.assertFalse(payload['store'])
        self.assertEqual(payload['reasoning']['effort'], 'medium')
        self.assertNotIn('SYNTHETIC_TEST_TOKEN', repr(client))

    @patch('urllib.request.build_opener')
    def test_plain_text_summary(self, opener):
        self.response(opener, [{'type': 'output_text', 'text': 'There are 8 rows.'}])
        client = OpenAILunaClient('SYNTHETIC_TEST_TOKEN')
        self.assertEqual(client.complete([{'role':'user','content':'Summarize'}]), 'There are 8 rows.')
        self.assertNotIn('text', json.loads(opener.return_value.open.call_args.args[0].data))

    @patch('urllib.request.build_opener')
    def test_incomplete_and_refusal_are_not_used_as_answers(self, opener):
        client = OpenAILunaClient('SYNTHETIC_TEST_TOKEN')
        for content, status in [([{'type':'output_text','text':'partial'}], 'incomplete'),
                                ([{'type':'refusal','refusal':'declined'}], 'completed')]:
            self.response(opener, content, status)
            with self.assertRaises(ModelError): client.complete([{'role':'user','content':'query'}])

    @patch('urllib.request.build_opener')
    def test_errors_do_not_expose_credentials_or_provider_body(self, opener):
        opener.return_value.open.side_effect = urllib.error.HTTPError(
            'https://api.openai.com/v1/responses', 401, 'PRIVATE_PROVIDER_BODY', {}, None)
        with self.assertRaises(ModelError) as failure:
            OpenAILunaClient('SYNTHETIC_TEST_TOKEN').complete([{'role':'user','content':'query'}])
        self.assertIn('401', str(failure.exception))
        self.assertNotIn('SYNTHETIC_TEST_TOKEN', str(failure.exception))
        self.assertNotIn('PRIVATE_PROVIDER_BODY', str(failure.exception))

    @patch.dict('os.environ', {'OPENAI_API_KEY':'SYNTHETIC_TEST_TOKEN'}, clear=True)
    def test_environment_selects_luna(self):
        self.assertIsInstance(OpenAICompatibleClient.from_env(), OpenAILunaClient)

    @patch('urllib.request.build_opener')
    def test_q2_uses_luna_and_keeps_real_source_citations(self, opener):
        def receive(request, **kwargs):
            payload = json.loads(request.data)
            concepts = json.loads(payload['input'][-1]['content'])
            choice = concepts['allowed_indices'][0]
            self.response(opener, [{'type':'output_text','text':json.dumps({'sentence_indices':[choice]})}])
            return opener.return_value.open.return_value
        opener.return_value.open.side_effect = receive
        agent = SupportAssistant(Path(__file__).resolve().parents[1] / 'docs/kb', OpenAILunaClient('SYNTHETIC_TEST_TOKEN'))
        result = agent.ask('How do I invite teammates?')
        self.assertEqual(result['mode'], 'model-selected-extractive')
        self.assertTrue(result['citations'])
        for citation in result['citations']:
            self.assertIn(citation['quote'], (agent.kb_path/citation['document']).read_text())

    def test_invalid_messages_make_no_request(self):
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ValueError):
                OpenAILunaClient('SYNTHETIC_TEST_TOKEN').complete([{'role':'user','content':123}])
            opener.assert_not_called()
