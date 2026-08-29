"""
Single source of truth for Gemini safety thresholds.

Both LLM paths must block the same content, but they need different shapes to
say so: the raw `genai` client takes a list of `types.SafetySetting`, while
`ChatGoogleGenerativeAI` takes a `{HarmCategory: HarmBlockThreshold}` dict. Both
are derived from `_THRESHOLDS` below, so the two paths cannot drift apart —
these were previously three hand-maintained copies (app/services/gemini.py,
app/services/image.py, app/services/llm.py) held together by "keep in sync"
comments.
"""

from __future__ import annotations

from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory

# The policy, stated once. BLOCK_LOW_AND_ABOVE is the strictest useful setting.
_THRESHOLDS: dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# Raw `genai` client form — passed as GenerateContentConfig(safety_settings=...).
SAFETY_SETTINGS: list[types.SafetySetting] = [
    types.SafetySetting(category=category, threshold=threshold)
    for category, threshold in _THRESHOLDS.items()
]

# LangChain form — passed to ChatGoogleGenerativeAI(safety_settings=...).
LANGCHAIN_SAFETY_SETTINGS: dict[HarmCategory, HarmBlockThreshold] = dict(_THRESHOLDS)
