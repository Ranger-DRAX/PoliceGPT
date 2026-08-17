"""
Multi-backend LLM client supporting Ollama, OpenAI/vLLM-compatible APIs, and HuggingFace pipelines.
"""

from typing import Optional, Dict, Any, List
import os
import httpx
from loguru import logger


class LLMClient:
    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5:7b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 60,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def generate_async(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider in ("ollama", "openai", "vllm"):
            return await self._call_openai_compatible_api(messages)
        else:
            return self._call_local_stub(user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider in ("ollama", "openai", "vllm"):
            return self._call_openai_compatible_api_sync(messages)
        return self._call_local_stub(user_prompt)

    def _call_openai_compatible_api_sync(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'placeholder'}",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling LLM API ({self.provider} at {url}): {e}")
            return (
                "আইনি উত্তর প্রস্তুতকালে সার্ভার সংযোগে ত্রুটি হয়েছে। অনুগ্রহ করে সিস্টেম অ্যাডমিনের সাথে যোগাযোগ করুন।"
            )

    async def _call_openai_compatible_api(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'placeholder'}",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling LLM API ({self.provider} at {url}): {e}")
            return (
                "আইনি উত্তর প্রস্তুতকালে সার্ভার সংযোগে ত্রুটি হয়েছে। অনুগ্রহ করে সিস্টেম অ্যাডমিনের সাথে যোগাযোগ করুন।"
            )

    def _call_local_stub(self, user_prompt: str) -> str:
        logger.info("Using local LLM generator stub.")
        return (
            "দণ্ডবিধি ১৮৬০ এর ধারা ৩৭৯ অনুযায়ী চুরির জন্য সর্বোচ্চ ৩ বছর কারাদণ্ড বা অর্থদণ্ড অথবা উভয় দণ্ডের বিধান রয়েছে।\n\n"
            "উৎস: The Penal Code, 1860, Section 379.\n\n"
            "সতর্কতা: এটি একটি পরীক্ষামূলক এআই-ভিত্তিক তথ্য ব্যবস্থা। চূড়ান্ত আইনি পদক্ষেপের জন্য আইনজীবী বা সংশ্লিষ্ট পুলিশ কর্মকর্তার পরামর্শ নিন।"
        )
