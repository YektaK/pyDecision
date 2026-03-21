###############################################################################

# Required Libraries
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

###############################################################################

# Load Libraries
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  

try:
    from google import genai
except Exception:
    genai = None 

###############################################################################

# Providers
@dataclass(frozen = True)
class ProviderSpec:
    provider:      str                      
    api_key_env:   str                   
    base_url:      Optional[str] = None      
    default_model: str = ""


PROVIDERS: Dict[str, ProviderSpec] = {
    # OpenAI
    "openai": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "OPENAI_API_KEY",
                                base_url      = None, 
                                default_model = "gpt-4o-mini",
                           ),

    # DeepSeek
    "deepseek": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "DEEPSEEK_API_KEY",
                                base_url      = "https://api.deepseek.com/v1",
                                default_model = "deepseek-chat",
                            ),

    # Qwen 
    "qwen-intl": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "DASHSCOPE_API_KEY",
                                base_url      = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                                default_model = "qwen-plus",
                             ),
    "qwen-us": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "DASHSCOPE_API_KEY",
                                base_url      = "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
                                default_model = "qwen-plus",
                            ),
    "qwen-cn": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "DASHSCOPE_API_KEY",
                                base_url      = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                default_model = "qwen-plus",
                            ),

    # Grok 
    "grok": ProviderSpec(
                                provider      = "openai_compat",
                                api_key_env   = "XAI_API_KEY",
                                base_url      = "https://api.x.ai/v1",
                                default_model = "grok-4",
                        ),

    # Gemini 
    "gemini": ProviderSpec(
                                provider      = "gemini",
                                api_key_env   = "GEMINI_API_KEY",
                                base_url      = None,
                                default_model = "gemini-3-flash-preview",
                          ),
}


###############################################################################

# Prompt Builders
def _df_to_text(df_like):
    if hasattr(df_like, "to_string"):
        return df_like.to_string(index = False)
    return str(df_like)

def build_prompt_from_table(table, query, context, char_limit = 4097):
    corpus = _df_to_text(table)
    prompt = f"{query}{context}:\n\n{corpus}\n"
    return prompt[:char_limit]


###############################################################################

# Callers
def _require_pkg(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def call_llm(prompt, *, provider_name = "openai", api_key = None, model = None, system = None, max_tokens = 2000, temperature = 0.8, timeout = None):
    spec       = PROVIDERS[provider_name]
    key        = api_key or os.getenv(spec.api_key_env, "")
    used_model = model or spec.default_model

    if spec.provider == "openai_compat":
        _require_pkg(OpenAI is not None, "Missing dependency: pip install -U openai")
        client_kwargs: Dict[str, Any] = {"api_key": key}
        if spec.base_url:
            client_kwargs["base_url"] = spec.base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        client = OpenAI(**client_kwargs)
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
                                                model       = used_model,
                                                messages    = messages,
                                                max_tokens  = max_tokens,
                                                temperature = temperature,
                                              )

        return resp.choices[0].message.content or ""

    if spec.provider == "gemini":
        _require_pkg(genai is not None, "Missing dependency: pip install -U google-genai")
        client     = genai.Client(api_key = key)
        final_text = prompt if not system else f"{system}\n\n{prompt}"
        response   = client.models.generate_content(
                                                        model    = used_model,
                                                        contents = final_text,
                                                    )
        return getattr(response, "text", "") or ""
    raise RuntimeError(f"Provider misconfigured: {provider_name}")


###############################################################################

# Function: Ask LLM Corr
def ask_llm_corr(table, char_limit = 4097, provider_name = "openai", api_key = None, query = "which methods are more similar?", model = None, max_tokens = 2000, temperature = 0.8, system = None):
    context = " knowing that for the given table, it shows the correlation values between MCDA methods"
    prompt  = build_prompt_from_table(table, query, context, char_limit)
    out     = call_llm(
                        prompt,
                        provider_name = provider_name,
                        api_key       = api_key,
                        model         = model,
                        max_tokens    = max_tokens,
                        temperature   = temperature,
                        system        = system,
                       )
    print("Number of Characters: " + str(len(prompt)))
    return out

# Function: Ask LLM Rank
def ask_llm_rank(table, char_limit = 4097, provider_name = "openai", api_key = None, query = "which methods are more similar?", model = None, max_tokens = 2000, temperature = 0.8, system = None):
    context = " knowing that for the given outranking table, the columns represent MCDA methods and the rows alternatives. Each cell indicates the rank of the alternatives (1 = 1st position, 2 = 2nd position, and so on) "
    prompt  = build_prompt_from_table(table, query, context, char_limit)
    out     = call_llm(
                        prompt,
                        provider_name = provider_name,
                        api_key       = api_key,
                        model         = model,
                        max_tokens    = max_tokens,
                        temperature   = temperature,
                        system        = system,
                       )
    print("Number of Characters: " + str(len(prompt)))
    return out

# Function: Ask LLM Weights
def ask_llm_weights(table, char_limit = 4097, provider_name = "openai", api_key = None, query = "which methods are more similar?", model = None, max_tokens = 2000, temperature = 0.8, system = None):
    context = " knowing that for the given table, the columns represent MCDA methods and the rows criterion. Each cell indicates the weight of each criterion (the higher the value, the most important is the criterion) calculate by each MCDA method"
    prompt  = build_prompt_from_table(table, query, context, char_limit)
    out     = call_llm(
                        prompt,
                        provider_name = provider_name,
                        api_key       = api_key,
                        model         = model,
                        max_tokens    = max_tokens,
                        temperature   = temperature,
                        system        = system,
                       )
    print("Number of Characters: " + str(len(prompt)))
    return out

###############################################################################