import io
import json
import unittest
from unittest.mock import patch
from desicrew.llm import OpenAICompatibleClient, ModelError
from desicrew.search import PublicDefinitionSearch

class AdapterTests(unittest.TestCase):
    def test_remote_http_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleClient('http://example.com/v1','demo')

    def test_docker_host_route_is_local(self):
        client=OpenAICompatibleClient("http://host.docker.internal:11434/v1","local-test-model")
        self.assertTrue(client.is_local)

    def test_embedded_credentials_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleClient('https://user:secret@example.com/v1','demo')

    @patch('urllib.request.build_opener')
    def test_configured_llm_request_and_json_output(self, urlopen):
        # Transport is mocked: this is not evidence of any live model call.
        raw=json.dumps({'choices':[{'message':{'content':'{"code":"result = len(df)"}'}}]}).encode()
        urlopen.return_value.open.return_value.__enter__.return_value.read.return_value=raw
        c=OpenAICompatibleClient('http://localhost:1234/v1','test-model')
        answer=c.complete([{'role':'user','content':'Return JSON'}],json_mode=True)
        self.assertEqual(json.loads(answer)['code'],'result = len(df)')
        request=urlopen.return_value.open.call_args.args[0]
        payload=json.loads(request.data)
        self.assertEqual(payload['response_format'],{'type':'json_object'})
        self.assertNotIn('Authorization',dict(request.header_items()))

    @patch('urllib.request.urlopen')
    def test_search_transmits_public_concept_only(self,urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value=json.dumps({'query':{'search':[{'title':'Inventory','snippet':'A <b>public</b> concept'}]}}).encode()
        search=PublicDefinitionSearch()
        result=search.search('inventory for secret customer 123456789')
        url=urlopen.call_args.args[0].full_url
        self.assertNotIn('secret',url)
        self.assertNotIn('123456789',url)
        self.assertEqual(result[0]['snippet'],'A public concept')

    @patch('urllib.request.urlopen')
    def test_unknown_concept_never_transmitted(self,urlopen):
        self.assertEqual(PublicDefinitionSearch().search('private employee report'),[])
        urlopen.assert_not_called()

    @patch('urllib.request.urlopen')
    def test_fifo_lifo_search_context_matches_label(self,urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value=b'{"query":{"search":[]}}'
        for term in ['fifo','lifo']:
            search=PublicDefinitionSearch()
            search.search(term+' inventory definition')
            self.assertIn(term.upper(),search.last_query)

    def test_model_redirects_fail_closed(self):
        from desicrew.llm import _NoModelRedirect
        import urllib.request
        handler = _NoModelRedirect()
        request = urllib.request.Request('https://provider.example/v1/chat/completions',data=b'{}',headers={'Authorization':'Bearer SYNTHETIC_TEST_TOKEN'})
        for target in ['http://other-host.example/collect','https://other-host.example/collect','https://provider.example/alternate']:
            self.assertIsNone(handler.redirect_request(request,None,302,'redirect',{},target))

    @patch.dict('os.environ',{},clear=True)
    def test_no_configuration_returns_none(self):
        self.assertIsNone(OpenAICompatibleClient.from_env())

if __name__ == '__main__':
    unittest.main()
