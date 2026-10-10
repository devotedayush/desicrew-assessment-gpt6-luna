"""Small explicit opt-in OpenAI-compatible adapter. Never called by tests or demos."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

LOCAL_MODEL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal", "host.containers.internal"}


class _NoModelRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward authorization or model data to a redirected destination."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ModelError(RuntimeError):
    """Safe model-provider error without credentials or provider response bodies."""

@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 45.0
    max_response_bytes: int = 2_000_000

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) provider URL")
        if parsed.scheme == "http" and parsed.hostname not in LOCAL_MODEL_HOSTS:
            raise ValueError("Use HTTPS for remote model providers")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Do not put credentials or query strings in LLM_BASE_URL")
        if not self.model.strip():
            raise ValueError("LLM_MODEL is required")
        self.base_url = self.base_url.rstrip("/")

    @property
    def is_local(self) -> bool:
        return urlparse(self.base_url).hostname in LOCAL_MODEL_HOSTS

    @classmethod
    def from_env(cls):
        """Only inspect the three named user-configured variables; no secret discovery."""
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            from desicrew.openai_luna import OpenAILunaClient
            return OpenAILunaClient(api_key=key)
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not base_url or not model:
            return None
        return cls(base_url=base_url, model=model, api_key=os.getenv("LLM_API_KEY", ""))

    def complete(self, messages, json_mode: bool = False) -> str:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 50:
            raise ValueError("Provide 1–50 conversation messages")
        clean = []
        for item in messages:
            if item.get("role") not in {"system", "user", "assistant"} or not isinstance(item.get("content"), str):
                raise ValueError("Each message needs a supported role and string content")
            clean.append({"role": item["role"], "content": item["content"]})
        payload = {"model": self.model, "messages": clean}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        encoded = json.dumps(payload).encode()
        if len(encoded) > 500_000:
            raise ValueError("Model request is too large")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + "/chat/completions", data=encoded, headers=headers)
        try:
            opener = urllib.request.build_opener(_NoModelRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ModelError("Model response exceeded the size limit")
            response = json.loads(raw)
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ModelError("Model response contained no text")
            return content
        except urllib.error.HTTPError as exc:
            raise ModelError(f"Model provider returned HTTP {exc.code}; check configuration and provider availability") from None
        except (urllib.error.URLError, TimeoutError):
            raise ModelError("Model provider was unreachable or timed out") from None
        except (KeyError, IndexError, json.JSONDecodeError):
            raise ModelError("Model provider returned an unexpected response format") from None
