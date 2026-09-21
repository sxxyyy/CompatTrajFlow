"""OpenAI-compatible MLLM client for trajectory anomaly detection.

Sends map images (base64-encoded PNGs) together with a system prompt to an
MLLM and parses the discrete ``Normal`` / ``Abnormal`` judgment from the
text response.

Responses are cached on disk so that interrupted runs can be resumed without
re-incurring API costs.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

from model.traj_mllm.prompts import JUDGMENT_PATTERN

logger = logging.getLogger(__name__)

_RESPONSE_CACHE_EXT = ".mllm_response.txt"


class MLLMClient:
    """Thin wrapper around an OpenAI-compatible chat-completions endpoint.

    Parameters
    ----------
    api_key : str
        API key.
    model_name : str
        Model identifier (e.g. ``"o4-mini"``).
    base_url : str
        API base URL.
    system_prompt : str
        System-level prompt sent with every request.
    cache_dir : str
        Directory where per-trajectory response text files are cached.
    max_retries : int
        Number of retries on transient API errors.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        system_prompt: str,
        cache_dir: str,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url
        self._system_prompt = system_prompt
        self._cache_dir = cache_dir
        self._max_retries = max_retries

        os.makedirs(self._cache_dir, exist_ok=True)

        # Lazy-import openai so the module is still importable without it.
        self._client: Any = None

    def predict(
        self, traj_index: int, user_content: str, image_paths: list[str],
        cache_scope: str = "",
    ) -> int:
        """Query the MLLM and return a binary prediction.

        Parameters
        ----------
        traj_index : int
            Trajectory index within its scope.
        user_content : str
            Short user-prompt text shown alongside the images.
        image_paths : list[str]
            Absolute paths to PNG images to include in the request.
        cache_scope : str
            Optional scope prefix to isolate caches (e.g. anomaly type name).

        Returns
        -------
        int
            ``0`` for *Normal*, ``1`` for *Abnormal*.
        """
        scope_prefix = f"{cache_scope}_" if cache_scope else ""
        cache_path = os.path.join(
            self._cache_dir, f"{scope_prefix}{traj_index}{_RESPONSE_CACHE_EXT}"
        )
        if os.path.exists(cache_path):
            logger.info("Trajectory %d: using cached MLLM response.", traj_index)
            with open(cache_path, "r", encoding="utf-8") as f:
                return self._parse_judgment(f.read())

        image_contents = self._encode_images(image_paths)
        if not image_contents:
            logger.warning(
                "Trajectory %d: no valid images found; defaulting to Normal.",
                traj_index,
            )
            return 0

        message_content: list[dict[str, Any]] = [{"type": "text", "text": user_content}]
        message_content.extend(image_contents)

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response_text = self._call_api(message_content)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                return self._parse_judgment(response_text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Trajectory %d: API attempt %d/%d failed: %s",
                    traj_index,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"All {self._max_retries} API attempts failed for trajectory "
            f"{traj_index}." + (f" Last error: {last_error}" if last_error else "")
        )

    def _get_client(self) -> Any:
        """Lazy-initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for MLLM inference. "
                    "Install it with: pip install openai"
                )

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=12000.0,
            )
        return self._client

    def _call_api(self, content: list[dict[str, Any]]) -> str:
        """Send a chat-completion request and return the full text response."""
        client = self._get_client()
        completion = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ],
            # Use streaming to handle long responses robustly.
            stream=True,
        )
        full_response: list[str] = []
        for chunk in completion:
            # Different SDK versions expose the content differently.
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                full_response.append(delta.content)
        return "".join(full_response)

    @staticmethod
    def _encode_images(paths: list[str]) -> list[dict[str, Any]]:
        """Base64-encode a list of image file paths.

        Returns a list of ``image_url`` content blocks suitable for the
        OpenAI chat-completions API.
        """
        blocks: list[dict[str, Any]] = []
        for p in paths:
            if not os.path.isfile(p):
                logger.warning("Image not found, skipping: %s", p)
                continue
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        return blocks

    @staticmethod
    def _parse_judgment(text: str) -> int:
        """Extract the binary label from the MLLM response.

        Returns ``0`` for *Normal*, ``1`` for *Abnormal*.  If the pattern
        cannot be found, defaults to ``0`` (Normal) with a warning.
        """
        match = JUDGMENT_PATTERN.search(text)
        if not match:
            logger.warning(
                "Could not find 'Final Judgment' in MLLM response; "
                "defaulting to Normal. Response preview: %.200s...",
                text,
            )
            return 0
        judgment = match.group(1).lower()
        return 1 if judgment == "abnormal" else 0
