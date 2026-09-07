"""
Replaceable LLM Client Interface and Google Gemini Implementation.
Provides BaseLLMClient interface, production GeminiClient (via REST/HTTPX with retries),
and MockLLMClient for reproducible offline testing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import time
from pydantic import BaseModel, Field
from loguru import logger
import httpx


class LLMResponse(BaseModel):
    """Normalized response container from any LLM provider."""
    text: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract base class for replaceable LLM model clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Generate response given a user prompt and optional system instructions."""
        pass


class GeminiClient(BaseLLMClient):
    """
    Production client for Google Gemini (gemini-2.5-flash, gemini-1.5-flash, etc.).
    Uses direct REST API via HTTPX with exponential backoff on HTTP 429 / 503.
    Requires GEMINI_API_KEY or GOOGLE_API_KEY in environment or constructor.
    """

    DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        timeout_sec: float = 30.0,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, max_retries)
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Execute generation call against Gemini API with retry mechanism."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key to GeminiClient."
            )

        endpoint = f"{self.DEFAULT_API_URL}/{self.model_name}:generateContent?key={self.api_key}"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.95,
            },
        }

        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        headers = {"Content-Type": "application/json"}
        t_start = time.perf_counter()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    resp = client.post(endpoint, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return LLMResponse(
                            text="কোন উত্তর তৈরি করা সম্ভব হয়নি (No candidates returned).",
                            model_name=self.model_name,
                            latency_ms=(time.perf_counter() - t_start) * 1000.0,
                            raw_response=data,
                        )

                    parts = candidates[0].get("content", {}).get("parts", [])
                    generated_text = "".join(p.get("text", "") for p in parts).strip()

                    usage = data.get("usageMetadata", {})
                    t_latency = (time.perf_counter() - t_start) * 1000.0

                    return LLMResponse(
                        text=generated_text,
                        model_name=self.model_name,
                        prompt_tokens=usage.get("promptTokenCount"),
                        completion_tokens=usage.get("candidatesTokenCount"),
                        latency_ms=round(t_latency, 2),
                        raw_response=data,
                    )

                elif resp.status_code in (429, 503):
                    # Rate limit or server overload -> wait with exponential backoff
                    wait_time = (2 ** attempt) + 0.5
                    logger.warning(
                        f"Gemini API returned HTTP {resp.status_code}. "
                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(wait_time)
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

                else:
                    error_msg = f"Gemini API error {resp.status_code}: {resp.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            except httpx.RequestError as e:
                last_error = e
                wait_time = (2 ** attempt) + 0.5
                logger.warning(f"Network error calling Gemini: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts. Last error: {last_error}")


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing and offline development.
    Produces deterministic answers with citation references.
    """

    def __init__(
        self,
        default_response: Optional[str] = None,
        model_name: str = "mock-gemini-flash",
    ):
        self.default_response = default_response
        self.model_name = model_name
        self.calls: list = []

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
        })

        if self.default_response:
            ans = self.default_response
        else:
            # Generate a realistic mock grounded answer
            ans = (
                "দণ্ডবিধি, ১৮৬০ এর ধারা ৩৭৯ অনুসারে, যে ব্যক্তি চুরি করে তাকে অনধিক "
                "৩ বছর পর্যন্ত যেকোনো মেয়াদের কারাদণ্ড, বা অর্থদণ্ড, বা উভয় দণ্ডে "
                "দণ্ডিত করা হবে [S1]।"
            )

        return LLMResponse(
            text=ans,
            model_name=self.model_name,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(ans) // 4,
            latency_ms=15.0,
        )
