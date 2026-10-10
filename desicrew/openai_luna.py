"""GPT-6 Luna adapter for the existing Q1/Q2 complete() interface."""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from desicrew.llm import ModelError, _NoModelRedirect


@dataclass
class OpenAILunaClient:
    api_key: str = field(repr=False)
    model: str = 'gpt-6-luna'
    base_url: str = 'https://api.openai.com/v1'
    timeout: float = 120.0
    max_response_bytes: int = 2_000_000

    def __post_init__(self):
        if not self.api_key.strip():
            raise ValueError('Set OPENAI_API_KEY or enter your key in the sidebar')
        if self.model != 'gpt-6-luna' or self.base_url != 'https://api.openai.com/v1':
            raise ValueError('Version 1 uses GPT-6 Luna at the official OpenAI endpoint')

    @property
    def is_local(self):
        return False

    def complete(self, messages, json_mode=False):
        if not isinstance(messages, list) or not 1 <= len(messages) <= 50:
            raise ValueError('Provide 1–50 conversation messages')
        clean = []
        for message in messages:
            if message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str):
                raise ValueError('Each message needs a supported role and string content')
            clean.append({'role': message['role'], 'content': message['content']})
        payload = {
            'model': self.model,
            'input': clean,
            'reasoning': {'effort': 'medium'},
            'max_output_tokens': 8192,
            'store': False,
        }
        if json_mode:
            payload['text'] = {'format': {'type': 'json_object'}}
        encoded = json.dumps(payload).encode('utf-8')
        if len(encoded) > 500_000:
            raise ValueError('Model request is too large')
        request = urllib.request.Request(
            self.base_url + '/responses', data=encoded,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key},
        )
        try:
            with urllib.request.build_opener(_NoModelRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ModelError('Model response exceeded the size limit')
            result = json.loads(raw)
            if result.get('status') != 'completed':
                raise ModelError('OpenAI did not complete the response; retry or check your output budget')
            parts = []
            for item in result.get('output', []):
                if item.get('type') != 'message':
                    continue
                for content in item.get('content', []):
                    if content.get('type') == 'refusal':
                        raise ModelError('OpenAI declined this request')
                    if content.get('type') == 'output_text' and isinstance(content.get('text'), str):
                        parts.append(content['text'])
            if not parts:
                raise ModelError('OpenAI returned no text')
            return ''.join(parts)
        except urllib.error.HTTPError as exc:
            raise ModelError(f'OpenAI returned HTTP {exc.code}; check your API key, model access, and billing') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ModelError('OpenAI was unreachable or timed out') from None
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            raise ModelError('OpenAI returned an unexpected response format') from None
