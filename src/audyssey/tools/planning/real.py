"""플래닝 담당 모듈의 기본 구현. 성능이 아니라 연결 확인용. 교체 가능."""

import base64
import json
import logging
from pathlib import Path
from typing import TypeVar

from langsmith import trace
from langsmith.run_helpers import tracing_context
from openai import APIStatusError
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import TypeAdapter, ValidationError

from audyssey.contracts import Model, PlanIssue, SourcePlan, Span, VisualHint
from audyssey.settings import Settings
from audyssey.tools.planning import llm


class _Sources(Model):
    sources: list[object]


class _Issues(Model):
    issues: list[PlanIssue]


class _Span(Model):
    start: float
    end: float


class _Hint(Model):
    time: float
    box: tuple[float, float, float, float]
    description: str


_Response = TypeVar("_Response", bound=Model)


def _messages(
    audio: Path, system: str, text: str
) -> list[ChatCompletionMessageParam]:
    content: list[ChatCompletionContentPartParam] = [
        {"type": "text", "text": text},
        {"type": "text", "text": "Original audio, starting at video time 0 seconds."},
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                "format": "wav",
            },
        },
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _request(
    settings: Settings,
    messages: list[ChatCompletionMessageParam],
    response_type: type[_Response],
) -> _Response:
    with llm.chat_client(settings) as client:
        for attempt in range(2):
            # Record model/usage only: the SDK wrapper would also trace media bytes.
            with trace(
                "planner_completion",
                run_type="llm",
                inputs={"model": settings.planner_model},
                metadata={
                    "ls_model_name": settings.planner_model,
                    "ls_model_type": "chat",
                },
            ) as run:
                with tracing_context(enabled=False, parent=False):
                    try:
                        response = client.chat.completions.create(
                            model=settings.planner_model,
                            messages=messages,
                            response_format={"type": "json_object"},
                        )
                    except APIStatusError as exc:
                        detail = exc.response.text.strip() or str(exc)
                        if settings.planner_api_key:
                            detail = detail.replace(settings.planner_api_key, "***")
                        raise ValueError(
                            f"planner HTTP {exc.status_code} "
                            f"(model={settings.planner_model}): {detail[:2000]}"
                        ) from None
                usage = response.usage
                run.end(
                    outputs={
                        "usage_metadata": {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    }
                    if usage
                    else {}
                )
            raw = response.choices[0].message.content if response.choices else None
            try:
                data = json.loads(raw or "")
                if isinstance(data, list):
                    # Gemini sometimes drops the {"sources": [...]} wrapper.
                    (field,) = response_type.model_fields
                    data = {field: data}
                return response_type.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Return valid JSON matching the schema.",
                        }
                    )
    excerpt = raw or ""
    if settings.planner_api_key:
        excerpt = excerpt.replace(settings.planner_api_key, "***")
    raise RuntimeError(f"planner returned invalid JSON: {excerpt[:500]}") from None


def _source(item: object, duration: float) -> SourcePlan:
    data = TypeAdapter(dict[str, object]).validate_python(item)
    if "spans" in data:
        if not isinstance(data["spans"], list):
            raise ValueError("spans must be a list")
        spans = []
        for value in data["spans"]:
            span = _Span.model_validate(value)
            start, end = (
                max(0, min(duration, span.start)),
                max(0, min(duration, span.end)),
            )
            if start < end:
                spans.append(Span(start=start, end=end))
        data["spans"] = spans
    if data.get("visual_hint") is not None:
        hint = _Hint.model_validate(data["visual_hint"])
        box = tuple(max(0, min(1, x)) for x in hint.box)
        data["visual_hint"] = (
            VisualHint(
                time=max(0, min(duration, hint.time)),
                box=(box[0], box[1], box[2], box[3]),
                description=hint.description,
            )
            if box[0] < box[2] and box[1] < box[3]
            else None
        )
    return SourcePlan.model_validate(data)


def plan(
    video: Path, audio: Path, duration: float, feedback: list[PlanIssue]
) -> list[SourcePlan]:
    settings = Settings()
    messages = _messages(
        audio,
        "You are an audio captioner identifying sound sources. "
        "Use short, plain captions for source names and descriptions, such as "
        "'dog barking' or 'trumpet'. Give enough information to identify the sound "
        "without detailed narration or unnecessary adjectives. "
        "Use common nouns only; do not use proper names, brands, character names, "
        "or specialized terminology. A source is one sound-producing entity. "
        "Do not split the same entity;"
        "Kinds: dialogue, sfx, music, ambience. Only original audio is available; "
        "identify sounds actually audible in the audio. Estimate span start/end "
        "in fractional seconds (for example, 1.25 to 2.70). Use the audio to "
        "estimate sound boundaries; do not invent unsupported precision. "
        "No visual information is available: always set visual_hint=null. "
        "Return JSON with a sources list. "
        "Each source follows this schema: "
        + json.dumps(SourcePlan.model_json_schema()),
        f"Video duration: {duration:g} seconds. Feedback: "
        + json.dumps([issue.model_dump() for issue in feedback]),
    )
    response = _request(settings, messages, _Sources)
    sources = []
    rejected = []
    for index, item in enumerate(response.sources):
        try:
            sources.append(_source(item, duration))
        except ValidationError as exc:
            # Report schema failures without echoing model output or credentials.
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'source'}: {error['type']}"
                for error in exc.errors(include_input=False, include_context=False)[:5]
            )
            rejected.append(f"source[{index}]: {details}")
        except ValueError:
            rejected.append(f"source[{index}]: invalid source fields")
    if not sources:
        detail = "; ".join(rejected[:5]) if rejected else "sources list was empty"
        raise RuntimeError(f"planner returned no valid sources: {detail}")
    return sources


def validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]:
    settings = Settings()
    messages = _messages(
        audio,
        "Check the source list against the original audio only. "
        "Report missing or inaudible sources, incorrect sound captions, and incorrect time spans. Verify short, plain captions without speculative narration. Use audio as evidence of sound presence and timing, require visual_hint=null because no visual information is available. "
        "Return JSON with an issues list (empty when none). Each issue follows: "
        + json.dumps(PlanIssue.model_json_schema()),
        "Sources: " + json.dumps([source.model_dump() for source in plans]),
    )
    try:
        return _request(settings, messages, _Issues).issues
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "plan validator returned invalid JSON twice; proceeding without issues"
        )
        return []
