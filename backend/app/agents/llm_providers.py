"""Free LLM backends for the strategist. All speak the OpenAI-compatible chat API, so one client
covers every one of them. No paid API is needed:

    groq        free tier, no card  (GROQ_API_KEY)        llama-3.3-70b-versatile, very fast
    gemini      free tier           (GEMINI_API_KEY)      gemini-2.5-flash
    openrouter  free ":free" models (OPENROUTER_API_KEY)  meta-llama/llama-3.3-70b-instruct:free
    ollama      runs on your laptop, offline, unlimited   qwen2.5:7b (any tool-capable model)
    mock        built-in, offline, deterministic (tests, demos without internet)

REO_LLM_PROVIDER picks one (default "auto": the first free key found, else mock).
REO_LLM_MODEL / REO_LLM_BASE_URL override the preset. Only the standard library is used.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Callable

PRESETS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", 6000),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash", 200000),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free", 20000),
    "ollama": ("http://localhost:11434/v1", "", "qwen2.5:7b", 10**9),
}


# if the configured model is retired, the first of these the account can see is used instead
FALLBACK_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                   "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct",
                   "qwen/qwen3-32b", "gemini-2.5-flash", "gemini-2.0-flash"]


class RateLimited(Exception):
    """Free-tier budget exhausted: skip the model this time (not an error)."""


class LLMClient:
    """Minimal OpenAI-compatible chat client with tool calling, a tokens-per-minute budget and
    automatic back-off on HTTP 429, so a free tier is never exceeded."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, tpm: int,
                 timeout_s: float = 6.0, post: Callable[[str, dict, dict, float], dict] | None = None,
                 get: Callable[[str, dict, float], dict] | None = None):
        self.name, self.base_url, self.api_key, self.model = name, base_url.rstrip("/"), api_key, model
        self.tpm, self.timeout_s = tpm, timeout_s
        self._post = post or _http_post
        self._get = get or _http_get
        self._resolved = False
        self._window: deque[tuple[float, int]] = deque()
        self._blocked_until = 0.0
        self.wait_for_budget = os.getenv("REO_LLM_WAIT", "0") == "1"

    def _budget_ok(self, estimate: int) -> bool:
        now = time.monotonic()
        while self._window and now - self._window[0][0] > 60:
            self._window.popleft()
        return now >= self._blocked_until and sum(t for _, t in self._window) + estimate <= self.tpm

    def chat(self, messages: list[dict], tools: list[dict], force_tool: str | None = None,
             estimate_tokens: int = 2000) -> tuple[dict, dict]:
        """Return (assistant_message, usage). Raises RateLimited when the free budget is used up."""
        if self.wait_for_budget:  # batch/CLI runs: wait for the free tier instead of skipping the model
            deadline = time.monotonic() + 70
            while not self._budget_ok(estimate_tokens) and time.monotonic() < deadline:
                time.sleep(1)
        if not self._budget_ok(estimate_tokens):
            raise RateLimited("free-tier budget")
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 1500,
            "tools": [{"type": "function", "function": t} for t in tools],
            "tool_choice": {"type": "function", "function": {"name": force_tool}} if force_tool else "auto",
        }
        if "gpt-oss" in self.model:  # reasoning model: keep thinking short so the JSON is never cut off
            body["reasoning_effort"] = "low"
        # some providers sit behind Cloudflare, which rejects Python's default "Python-urllib" agent
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "renewable-energy-orchestrator/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            try:
                data = self._post(self.base_url + "/chat/completions", body, headers, self.timeout_s)
            except urllib.error.HTTPError as e:
                if e.code == 404 and not self._resolved and self._switch_model(headers):
                    body["model"] = self.model
                    data = self._post(self.base_url + "/chat/completions", body, headers, self.timeout_s)
                else:
                    raise
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._blocked_until = time.monotonic() + 30
                raise RateLimited("HTTP 429 from provider") from e
            try:
                detail = e.read().decode(errors="replace")[:200]
            except Exception:
                detail = ""
            raise RuntimeError(f"HTTP {e.code} from {self.name}: {detail}") from e
        u = data.get("usage") or {}
        used = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
        self._window.append((time.monotonic(), used or estimate_tokens))
        msg = data["choices"][0]["message"]
        return msg, {"in": int(u.get("prompt_tokens", 0)), "out": int(u.get("completion_tokens", 0))}


    def available_models(self, headers: dict | None = None) -> list[str]:
        headers = headers or ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        headers = {**headers, "User-Agent": "renewable-energy-orchestrator/1.0"}
        data = self._get(self.base_url + "/models", headers, self.timeout_s)
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]

    def _switch_model(self, headers: dict) -> bool:
        """Model retired or not enabled: pick the best one this account can use."""
        self._resolved = True
        try:
            ids = self.available_models(headers)
        except Exception:
            return False
        if self.model in ids:
            return False
        pick = next((m for m in FALLBACK_MODELS if m in ids), None)
        if pick:
            self.model = pick
        return bool(pick)


def _http_get(url: str, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tool_calls(msg: dict) -> list[tuple[str, dict, str]]:
    """[(name, args, call_id)] from an assistant message; tolerant of small models that put the
    JSON in plain text instead of a tool call."""
    out = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            args = json.loads(args) if isinstance(args, str) else dict(args)
        except (ValueError, TypeError):
            args = {}
        out.append((fn.get("name", ""), args if isinstance(args, dict) else {}, c.get("id") or f"call_{i}"))
    if not out and msg.get("content"):
        m = re.search(r"\{.*\}", msg["content"], re.S)
        if m:
            try:
                out.append(("submit_directive", json.loads(m.group(0)), "text"))
            except ValueError:
                pass
    return out


def make_client() -> LLMClient | None:
    """None means: use the offline mock strategist."""
    choice = os.getenv("REO_LLM_PROVIDER", "auto").lower()
    if choice == "mock":
        return None
    if choice == "auto":
        choice = next((n for n in ("groq", "gemini", "openrouter") if os.getenv(PRESETS[n][1])), "mock")
        if choice == "mock":
            return None
    if choice not in PRESETS:
        raise KeyError(f"Unknown REO_LLM_PROVIDER '{choice}'. Use one of {sorted(PRESETS)} or mock")
    base, key_env, model, tpm = PRESETS[choice]
    return LLMClient(choice, os.getenv("REO_LLM_BASE_URL", base), os.getenv(key_env, "") if key_env else "",
                     os.getenv("REO_LLM_MODEL", model), int(os.getenv("REO_LLM_TPM", tpm)),
                     float(os.getenv("REO_LLM_TIMEOUT", "6")))
