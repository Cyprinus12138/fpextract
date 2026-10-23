"""Provider-agnostic vision-LLM client.

Two prompting strategies, both available on every provider:

* **tool mode** — the extraction JSON Schema is forced as a tool/function
  call, so the model cannot return malformed JSON. Supported by Anthropic
  and OpenAI-style function-calling endpoints. The only mode with a
  **batch API** (50% price, async).
* **json mode** — a plain-JSON prompt for VLMs that do not support function
  calling (Qwen3-VL / Qwen2.5-VL on vLLM or DashScope, LLaVA, InternVL, …).
  Handles ``<think>…</think>`` chain-of-thought and ``reasoning_content``
  fields, retries on truncated JSON, and converts bounding boxes from the
  model's coordinate system to ``crop_frac`` fractions.

Providers::

    anthropic  — ANTHROPIC_API_KEY
    openai     — OPENAI_API_KEY + optional base_url for any OpenAI-compatible
                 endpoint:  Ollama    http://localhost:11434/v1
                            vLLM      http://localhost:8000/v1
                            LM Studio http://localhost:1234/v1
                            DashScope https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                            Azure     https://<res>.openai.azure.com/openai/deployments/<dep>
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from typing import Any

from fpextract.images import sent_dims
from fpextract.llm.coords import normalise_record_bboxes

TOOL_NAME = "record_extraction"
TOOL_DESCRIPTION = "Record the structured extraction for this floor plan image."
USER_INSTRUCTION = "Extract this floor plan. Call record_extraction exactly once."

PROVIDERS = ("anthropic", "openai")

# Short aliases → model ids. Verified against the upstream pipeline this
# package was extracted from; override with a full model id at any time.
ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "haiku": "claude-haiku-4-5-20251001",
}
OPENAI_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "gpt-4o",
    "opus": "gpt-4o",
    "haiku": "gpt-4o-mini",
}


def resolve_model(provider: str, model: str) -> str:
    """Expand a short alias to a full model id for the given provider."""
    if provider == "anthropic":
        return ANTHROPIC_MODEL_ALIASES.get(model, model)
    return OPENAI_MODEL_ALIASES.get(model, model)


def api_key_from_env(provider: str, base_url: str | None = None) -> str:
    """Read the provider's API key from the environment.

    For OpenAI-compatible local endpoints (Ollama, vLLM, LM Studio) the SDK
    insists on a non-empty key the server never checks, so a placeholder is
    substituted when ``base_url`` is set and no key is exported.
    """
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    key = os.environ.get(env_var)
    if key:
        return key
    if provider == "openai" and base_url:
        return "not-required"
    raise SystemExit(f"{env_var} not set in environment.")


# ---------------------------------------------------------------------------
# Packaged prompt + schema
# ---------------------------------------------------------------------------

def load_prompt(name: str = "extract") -> str:
    return resources.files("fpextract.llm").joinpath(f"prompts/{name}.md").read_text(encoding="utf-8")


def load_extraction_schema() -> dict:
    """The per-image extraction schema with top-level ``$``-keys stripped
    (tool ``input_schema`` wants the bare JSON Schema body)."""
    raw = resources.files("fpextract.llm").joinpath("schema/extraction.schema.json").read_text(encoding="utf-8")
    schema = json.loads(raw)
    return {k: v for k, v in schema.items() if not k.startswith("$")}


# ---------------------------------------------------------------------------
# Batch value types
# ---------------------------------------------------------------------------

@dataclass
class BatchRequest:
    """One image to extract in a batch run. Provider-agnostic."""
    custom_id: str
    image_bytes: bytes
    media_type: str
    model: str
    system_prompt: str
    tool_schema: dict


@dataclass
class BatchStatus:
    """Normalised batch status across providers."""
    status: str          # "pending" | "completed" | "failed"
    raw_status: str      # provider's native status string
    counts: dict         # {processing, succeeded, errored, total} where available


@dataclass
class BatchResult:
    """One result from a batch run: a parsed record or an error."""
    custom_id: str
    succeeded: bool
    record: dict | None
    usage: dict | None   # {"input_tokens", "output_tokens"}
    error: str | None


# ---------------------------------------------------------------------------
# JSON-mode helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_json_reply(raw: str) -> dict:
    """Tolerant JSON parse: strips thinking tags and code fences, then takes
    the outermost ``{…}`` so a trailing remark does not break the parse."""
    s = strip_thinking(raw)
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE).strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    return json.loads(s)


JSON_SYSTEM = (
    "You are a precise floor-plan data extractor. "
    "Return valid JSON only — no prose, no markdown fences."
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified interface. Instantiate via :func:`make_client`."""

    provider: str = ""

    # --- tool mode -------------------------------------------------------
    def extract_tool(self, image_bytes: bytes, media_type: str, model: str,
                     system_prompt: str, tool_schema: dict) -> tuple[dict, dict]:
        raise NotImplementedError

    # --- json mode -------------------------------------------------------
    def complete_json_raw(self, image_bytes: bytes, media_type: str, model: str,
                          system: str, prompt: str, max_tokens: int,
                          no_think: bool = False) -> tuple[str, dict]:
        raise NotImplementedError

    def complete_json(self, image_bytes: bytes, media_type: str, model: str,
                      prompt: str, system: str = JSON_SYSTEM, max_tokens: int = 16384,
                      no_think: bool = False, retries: int = 3) -> tuple[dict, dict]:
        """One plain-JSON call with retry on parse failure (truncated output
        from an exhausted thinking budget, transient rate limiting)."""
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            if attempt > 1:
                time.sleep(2 ** (attempt - 1))
            raw, usage = self.complete_json_raw(image_bytes, media_type, model, system,
                                                prompt, max_tokens, no_think=no_think)
            try:
                return parse_json_reply(raw), usage
            except (json.JSONDecodeError, ValueError) as e:
                last_err = RuntimeError(
                    f"JSON parse failed (attempt {attempt}/{retries}): {e}; raw={raw[:300]!r}")
        assert last_err is not None
        raise last_err

    def extract_json(self, image_bytes: bytes, media_type: str, model: str, prompt: str,
                     coord: str = "0-1000", **kw) -> tuple[dict, dict]:
        """JSON-mode extraction with bbox → ``crop_frac`` normalisation."""
        record, usage = self.complete_json(image_bytes, media_type, model, prompt, **kw)
        h, w = sent_dims(image_bytes)
        normalise_record_bboxes(record, coord, h, w)
        return record, usage

    # --- batch (tool mode only) -----------------------------------------
    def submit_batch(self, requests: list[BatchRequest]) -> str:
        raise NotImplementedError

    def poll_batch(self, batch_id: str) -> BatchStatus:
        raise NotImplementedError

    def fetch_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        raise NotImplementedError

    @property
    def effective_base_url(self) -> str | None:
        sdk = getattr(self, "_client", None)
        return str(getattr(sdk, "base_url", None)) if sdk is not None else None


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, timeout: float = 120.0) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic") from e
        self._client = _anthropic.Anthropic(api_key=api_key, timeout=timeout)

    @staticmethod
    def _image_block(image_bytes: bytes, media_type: str) -> dict:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    def _tool_params(self, image_bytes, media_type, model, system_prompt, tool_schema) -> dict:
        return {
            "model": model,
            "max_tokens": 8192,
            "system": system_prompt,
            "tools": [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": tool_schema}],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "messages": [{"role": "user", "content": [
                self._image_block(image_bytes, media_type),
                {"type": "text", "text": USER_INSTRUCTION},
            ]}],
        }

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage", None)
        return {"input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None)}

    def extract_tool(self, image_bytes, media_type, model, system_prompt, tool_schema):
        resp = self._client.messages.create(
            **self._tool_params(image_bytes, media_type, model, system_prompt, tool_schema))
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                return dict(block.input), self._usage(resp)
        raise RuntimeError(f"model did not call {TOOL_NAME}; stop_reason={resp.stop_reason}")

    def complete_json_raw(self, image_bytes, media_type, model, system, prompt, max_tokens,
                          no_think=False):
        resp = self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": [
                self._image_block(image_bytes, media_type),
                {"type": "text", "text": prompt},
            ]}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return raw, self._usage(resp)

    # batch — Anthropic Message Batches API
    def submit_batch(self, requests):
        payload = [{"custom_id": r.custom_id,
                    "params": self._tool_params(r.image_bytes, r.media_type, r.model,
                                                r.system_prompt, r.tool_schema)}
                   for r in requests]
        return self._client.messages.batches.create(requests=payload).id

    def poll_batch(self, batch_id):
        b = self._client.messages.batches.retrieve(batch_id)
        status = "completed" if b.processing_status == "ended" else "pending"
        rc = b.request_counts
        counts = {k: getattr(rc, k, 0) for k in ("processing", "succeeded", "errored", "canceled", "expired")} if rc else {}
        return BatchStatus(status=status, raw_status=b.processing_status, counts=counts)

    def fetch_batch_results(self, batch_id):
        for r in self._client.messages.batches.results(batch_id):
            cid, kind = r.custom_id, r.result.type
            if kind != "succeeded":
                yield BatchResult(cid, False, None, None, f"{kind}: {getattr(r.result, 'error', None)}")
                continue
            msg = r.result.message
            record = None
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                    record = dict(block.input)
                    break
            if record is None:
                yield BatchResult(cid, False, None, None,
                                  f"model did not call {TOOL_NAME}; stop_reason={msg.stop_reason}")
            else:
                yield BatchResult(cid, True, record, self._usage(msg), None)


# ---------------------------------------------------------------------------
# OpenAI and OpenAI-compatible
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120.0) -> None:
        try:
            import openai as _openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai  (or: pip install fpextract[openai])") from e
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**kwargs)

    @staticmethod
    def _data_url(image_bytes: bytes, media_type: str) -> str:
        return f"data:{media_type};base64," + base64.standard_b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _dump(resp: Any) -> str:
        try:
            if hasattr(resp, "model_dump"):
                return json.dumps(resp.model_dump(), default=str, ensure_ascii=False)
            return json.dumps(resp, default=str, ensure_ascii=False)
        except Exception:
            return repr(resp)

    @staticmethod
    def _explain_or_raise(e: Exception) -> None:
        """Translate the common OpenAI-compatible failure modes into
        actionable messages; re-raise anything unrecognised."""
        msg = str(e)
        if "image_url" in msg and ("unknown variant" in msg or "expected" in msg):
            raise RuntimeError(
                "The endpoint rejected the image_url content part: the model or server "
                "does not support vision via /v1/chat/completions. Use a vision-capable "
                "model (gpt-4o, llava, qwen-vl, internvl, …) on a multimodal server. "
                "Original error: " + msg) from e
        if "tool_choice" in msg or ("function" in msg.lower() and "unknown" in msg.lower()):
            raise RuntimeError(
                "The endpoint rejected tool_choice / function calling. Either use a server "
                "that supports OpenAI-style function calling, or switch to --mode json. "
                "Original error: " + msg) from e
        raise e

    @staticmethod
    def _usage(resp) -> dict:
        usage = {"input_tokens": None, "output_tokens": None}
        if u := getattr(resp, "usage", None):
            usage["input_tokens"] = getattr(u, "prompt_tokens", None)
            usage["output_tokens"] = getattr(u, "completion_tokens", None)
        return usage

    def _tool_body(self, image_bytes, media_type, model, system_prompt, tool_schema) -> dict:
        return {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url(image_bytes, media_type)}},
                    {"type": "text", "text": USER_INSTRUCTION},
                ]},
            ],
            "tools": [{"type": "function", "function": {
                "name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": tool_schema}}],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }

    def extract_tool(self, image_bytes, media_type, model, system_prompt, tool_schema):
        try:
            resp = self._client.chat.completions.create(
                **self._tool_body(image_bytes, media_type, model, system_prompt, tool_schema))
        except Exception as e:
            self._explain_or_raise(e)
            raise  # pragma: no cover
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError(f"server returned no choices (model={model}); raw: {self._dump(resp)}")
        choice = choices[0]
        for tc in getattr(getattr(choice, "message", None), "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is None or getattr(fn, "name", None) != TOOL_NAME:
                continue
            if args := getattr(fn, "arguments", None):
                return json.loads(args), self._usage(resp)
        raise RuntimeError(f"model did not call {TOOL_NAME}; finish_reason="
                           f"{getattr(choice, 'finish_reason', None)!r}; raw: {self._dump(resp)}")

    def complete_json_raw(self, image_bytes, media_type, model, system, prompt, max_tokens,
                          no_think=False):
        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url(image_bytes, media_type)}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        }
        if no_think:
            # vLLM / Qwen3: disable hidden chain-of-thought. Ignored by servers
            # that do not know the field.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            self._explain_or_raise(e)
            raise  # pragma: no cover
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError(f"server returned no choices (model={model}); raw: {self._dump(resp)}")
        msg = choices[0].message
        raw = ""
        # Some reasoning deployments leave `content` empty and put the answer
        # in `reasoning` / `reasoning_content`.
        for field in ("content", "reasoning", "reasoning_content"):
            candidate = (getattr(msg, field, None) or "").strip()
            if candidate:
                raw = candidate
                break
        return raw, self._usage(resp)

    # batch — OpenAI Batch API (JSONL upload + batch referencing the file)
    def submit_batch(self, requests):
        lines = [json.dumps({"custom_id": r.custom_id, "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": self._tool_body(r.image_bytes, r.media_type, r.model,
                                                     r.system_prompt, r.tool_schema)},
                            ensure_ascii=False) for r in requests]
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        file_obj = self._client.files.create(file=("batch.jsonl", jsonl_bytes, "application/jsonl"),
                                             purpose="batch")
        batch = self._client.batches.create(input_file_id=file_obj.id,
                                            endpoint="/v1/chat/completions",
                                            completion_window="24h")
        return batch.id

    def poll_batch(self, batch_id):
        b = self._client.batches.retrieve(batch_id)
        if b.status in {"completed"}:
            status = "completed"
        elif b.status in {"failed", "expired", "cancelled"}:
            status = "failed"
        else:
            status = "pending"
        rc = b.request_counts
        counts = {k: getattr(rc, k, 0) for k in ("completed", "failed", "total")} if rc else {}
        return BatchStatus(status=status, raw_status=b.status, counts=counts)

    def fetch_batch_results(self, batch_id):
        b = self._client.batches.retrieve(batch_id)
        for file_id in (getattr(b, "output_file_id", None), getattr(b, "error_file_id", None)):
            if not file_id:
                continue
            content = self._client.files.content(file_id)
            text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = row["custom_id"]
                if row.get("error"):
                    yield BatchResult(cid, False, None, None, f"{row['error']}")
                    continue
                resp = row.get("response") or {}
                if resp.get("status_code") != 200:
                    yield BatchResult(cid, False, None, None,
                                      f"HTTP {resp.get('status_code')}; raw row: {json.dumps(row, default=str)}")
                    continue
                body = resp.get("body") or {}
                choices = body.get("choices") or []
                if not choices:
                    yield BatchResult(cid, False, None, None,
                                      f"no choices in batch response; raw row: {json.dumps(row, default=str)}")
                    continue
                choice = choices[0]
                record = None
                for tc in (choice.get("message") or {}).get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name") == TOOL_NAME and fn.get("arguments"):
                        record = json.loads(fn["arguments"])
                        break
                if record is None:
                    yield BatchResult(cid, False, None, None,
                                      f"model did not call {TOOL_NAME}; finish_reason={choice.get('finish_reason')!r}")
                else:
                    u = body.get("usage") or {}
                    yield BatchResult(cid, True, record,
                                      {"input_tokens": u.get("prompt_tokens"),
                                       "output_tokens": u.get("completion_tokens")}, None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(provider: str, api_key: str, base_url: str | None = None,
                timeout: float = 120.0) -> LLMClient:
    """Return a provider client with the unified :class:`LLMClient` interface.

    ``timeout`` covers connect + read per request. 120 s is generous for
    cloud APIs; drop to 30–60 s for local endpoints to fail fast.
    """
    if provider == "anthropic":
        if base_url:
            raise ValueError("base_url is not supported for the anthropic provider")
        return AnthropicClient(api_key=api_key, timeout=timeout)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)
    raise ValueError(f"unknown provider {provider!r} — choose one of {PROVIDERS}")
