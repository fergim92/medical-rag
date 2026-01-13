"""
LLM client supporting multiple providers.
"""

import os
from typing import Optional
import requests


class LLMClient:
    """
    LLM client supporting multiple providers.
    Supported: Groq API (cloud), Ollama (local)
    """

    ENDPOINTS = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "ollama": "http://localhost:11434/api/chat"
    }

    MODELS = {
        "groq": "llama-3.1-70b-versatile",
        "ollama": "llama3.1"
    }

    def __init__(self, provider: str = "groq", api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider ('groq' or 'ollama')
            api_key: API key for cloud providers
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if provider == "groq" and not self.api_key:
            print("Warning: GROQ_API_KEY not set. Set it or use 'ollama' provider.")

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0-1)

        Returns:
            Generated text response
        """
        if self.provider == "groq":
            return self._groq_generate(prompt, system_prompt, temperature)
        elif self.provider == "ollama":
            return self._ollama_generate(prompt, system_prompt, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _groq_generate(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate using Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.MODELS["groq"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        response = requests.post(
            self.ENDPOINTS["groq"],
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _ollama_generate(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate using local Ollama."""
        data = {
            "model": self.MODELS["ollama"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": temperature}
        }

        response = requests.post(
            self.ENDPOINTS["ollama"],
            json=data,
            timeout=60
        )
        response.raise_for_status()

        return response.json()["message"]["content"]
