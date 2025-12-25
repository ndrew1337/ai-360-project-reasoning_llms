import os
import sys
import time
import faulthandler

import openai
import backoff
import requests

completion_tokens = 0
prompt_tokens = 0

# ---- env knobs ----
OPENAI_CONNECT_TIMEOUT = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "10"))
OPENAI_READ_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_BACKOFF_MAX = float(os.getenv("OPENAI_BACKOFF_MAX", "120"))

def _on_backoff(details):
    e = details.get("exception")
    print(
        f"[OPENAI] backoff tries={details.get('tries')} "
        f"wait={details.get('wait', 0):.2f}s "
        f"err={type(e).__name__}: {e}",
        file=sys.stderr,
        flush=True,
    )

def _giveup(e: Exception) -> bool:
    # не ретраим то, что "не починится ожиданием"
    non_retry = (
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
        openai.error.PermissionError,
    )
    return isinstance(e, non_retry)

@backoff.on_exception(
    backoff.expo,
    (openai.error.OpenAIError, requests.exceptions.RequestException),
    max_time=OPENAI_BACKOFF_MAX,
    on_backoff=_on_backoff,
    giveup=_giveup,
)
def completions_with_backoff(**kwargs):
    # важный момент: tuple timeout (connect, read)
    kwargs.setdefault("request_timeout", (OPENAI_CONNECT_TIMEOUT, OPENAI_READ_TIMEOUT))

    # опционально: дамп стека только если "реально долго"
    if os.getenv("TOT_DEBUG_API") == "1":
        faulthandler.dump_traceback_later(
            int(OPENAI_READ_TIMEOUT) + 5,
            repeat=False,
            file=sys.stderr,
        )

    try:
        return openai.ChatCompletion.create(**kwargs)
    finally:
        if os.getenv("TOT_DEBUG_API") == "1":
            faulthandler.cancel_dump_traceback_later()

# ---- OpenAI init ----
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key:
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set", file=sys.stderr, flush=True)

api_base = os.getenv("OPENAI_API_BASE", "")
if api_base:
    print(f"Warning: OPENAI_API_BASE is set to {api_base}", file=sys.stderr, flush=True)
    openai.api_base = api_base

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens

    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt

        t0 = time.time()
        if os.getenv("TOT_DEBUG_API") == "1":
            print(f"[OPENAI] request model={model} n={cnt} max_tokens={max_tokens} stop={repr(stop)}", flush=True)

        res = completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
        )

        if os.getenv("TOT_DEBUG_API") == "1":
            dt = time.time() - t0
            usage = getattr(res, "usage", None)
            u = ""
            if usage is not None:
                u = f" prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
            print(f"[OPENAI] response in {dt:.2f}s{u}", flush=True)

        outputs.extend([choice.message.content for choice in res.choices])

        # usage может отсутствовать у некоторых прокси/старых версий — страхуемся
        if getattr(res, "usage", None) is not None:
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens

    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    # оставляю твою логику как есть
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    elif backend == "gpt-4.1-nano":
        cost = (prompt_tokens / 1000) * 0.0001 + (completion_tokens / 1000) * 0.0004
    elif backend == "gpt-4.1-mini":
        cost = (prompt_tokens / 1000) * 0.0004 + (completion_tokens / 1000) * 0.0016
    elif backend == "gpt-4o-mini":
        cost = (prompt_tokens / 1000) * 0.00015 + (completion_tokens / 1000) * 0.00060
    elif backend == "gpt-5-nano":
        cost = (prompt_tokens / 1000) * 0.00005 + (completion_tokens / 1000) * 0.0004
    elif backend == "gpt-5-mini":
        cost = (prompt_tokens / 1000) * 0.00025 + (completion_tokens / 1000) * 0.002
    else:
        cost = 0.0

    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
