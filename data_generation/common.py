"""Shared, dependency-light helpers for the ARIVE data-generation stages."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list or a JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must contain a list of JSON objects")
    return rows


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def dump_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_id(record: dict[str, Any]) -> Any:
    for key in ("id", "question_id"):
        if key in record:
            return record[key]
    raise KeyError("record has neither 'id' nor 'question_id'")


def report_text(record: dict[str, Any]) -> str:
    for key in ("report", "findings"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if not isinstance(turn, dict):
                continue
            value = turn.get("value")
            if isinstance(value, str):
                return value
    raise KeyError("record has no report text in report/findings/conversations")


def reasoning_text(record: dict[str, Any]) -> str:
    for key in ("reasoning", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", [])
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                assistant_text = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                continue
            match = re.search(r"<think>(.*?)</think>", assistant_text, re.DOTALL)
            return (match.group(1) if match else assistant_text).strip()
    raise KeyError("record has no reasoning text")


def image_list(record: dict[str, Any]) -> list[str]:
    value = record.get("image")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError("record['image'] must be a path string or a list of path strings")


def study_id_from_path(image_path: str) -> str:
    """Extract a MIMIC-style ``s########`` study id without assuming a root path."""
    parts = Path(image_path.replace("\\", "/")).parts
    for part in parts:
        if re.fullmatch(r"s\d+", part):
            return part
    raise ValueError(f"Cannot infer study id from image path: {image_path}")


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def context_value(context: dict[str, Any], key: str) -> str:
    value = context.get(key, "")
    return "" if is_missing_scalar(value) else str(value).strip()


def clinical_context_text(context: dict[str, Any]) -> str:
    """Format clinical evidence for reasoning generation, before SFT conversion."""
    return "\n".join(
        f"{key}: {value}" for key in ("Indication", "History", "Technique", "Comparison")
        if (value := context_value(context, key))
    )


def add_generation_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path | None = None,
) -> None:
    parser.add_argument("--model", required=True, help="Local path or model id for GPT-OSS.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        required=default_output is None,
    )
    parser.add_argument("--start", type=int, default=0, help="Inclusive record offset.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive record offset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")


class HarmonyGenerator:
    """Small wrapper around vLLM's token-prompt API and GPT-OSS Harmony."""

    def __init__(
        self,
        *,
        model: str,
        max_model_len: int,
        max_tokens: int,
        temperature: float,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        seed: int | None,
        developer_instruction: str = "You are an expert radiologist.",
    ) -> None:
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            from vllm import LLM, SamplingParams, TokensPrompt
        except ImportError as exc:
            raise RuntimeError(
                "Prompt generation requires vllm and openai-harmony; "
                "install requirements/generation.txt"
            ) from exc

        if max_model_len <= 0 or max_tokens <= 0 or tensor_parallel_size <= 0:
            raise ValueError("model/token lengths and tensor parallel size must be positive")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu memory utilization must be in (0, 1]")

        self._tokens_prompt = TokensPrompt
        self.developer_instruction = developer_instruction
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_token_ids": self.encoding.stop_tokens_for_assistant_actions(),
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        self.sampling = SamplingParams(**sampling_kwargs)

    def _render(self, user_prompt: str) -> list[int]:
        from openai_harmony import Conversation, Message, Role

        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, "Reasoning: high"),
                Message.from_role_and_content(Role.DEVELOPER, self.developer_instruction),
                Message.from_role_and_content(Role.USER, user_prompt),
            ]
        )
        return self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    @staticmethod
    def _last_text(entries: Iterable[Any], fallback: str) -> str:
        for entry in reversed(list(entries)):
            value = entry.to_dict() if hasattr(entry, "to_dict") else entry
            if not isinstance(value, dict):
                continue
            content = value.get("content", [])
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for item in reversed(content):
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        if item["text"].strip():
                            return item["text"].strip()
        return fallback.strip()

    def generate(self, prompts: Sequence[str]) -> list[str]:
        from openai_harmony import Role

        token_prompts = [
            self._tokens_prompt(prompt_token_ids=self._render(prompt)) for prompt in prompts
        ]
        outputs = self.llm.generate(prompts=token_prompts, sampling_params=self.sampling)
        texts: list[str] = []
        for output in outputs:
            generated = output.outputs[0]
            try:
                entries = self.encoding.parse_messages_from_completion_tokens(
                    generated.token_ids, Role.ASSISTANT
                )
                texts.append(self._last_text(entries, generated.text))
            except Exception:
                texts.append(generated.text.strip())
        return texts


def run_harmony_stage(
    *,
    records: Sequence[dict[str, Any]],
    prompt_builder: Callable[[dict[str, Any]], str],
    args: argparse.Namespace,
    output_transform: Callable[[str], str] | None = None,
    developer_instruction: str = "You are an expert radiologist.",
    record_output_transform: Callable[[dict[str, Any], str], str] | None = None,
    raw_output_path: Path | None = None,
    output_fields: dict[str, Any] | None = None,
    saved_drafts: dict[str, str] | None = None,
    review_prompt_builder: Callable[[dict[str, Any], str], str] | None = None,
    review_developer_instruction: str | None = None,
    review_output_transform: Callable[[dict[str, Any], str, str], str] | None = None,
    review_raw_output_path: Path | None = None,
) -> None:
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("invalid --start/--end range")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    selected = list(records[args.start : args.end])
    if not selected:
        raise ValueError("selected input range is empty")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite to replace it"
        )
    if raw_output_path is not None and raw_output_path.exists() and not args.overwrite:
        raise FileExistsError(f"raw output already exists: {raw_output_path}")
    if review_raw_output_path is not None and review_raw_output_path.exists() and not args.overwrite:
        raise FileExistsError(f"review output already exists: {review_raw_output_path}")

    generator = HarmonyGenerator(
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        developer_instruction=developer_instruction,
    )
    results: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    review_results: list[dict[str, Any]] = []
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        generated = ([saved_drafts[str(record_id(record))] for record in batch]
                     if saved_drafts is not None
                     else generator.generate([prompt_builder(record) for record in batch]))
        if len(generated) != len(batch):
            raise RuntimeError(
                f"generation result count mismatch: expected={len(batch)}, got={len(generated)}"
            )
        if raw_output_path is not None:
            raw_results.extend({"id": record_id(record), "response": text}
                               for record, text in zip(batch, generated))
            dump_json(raw_output_path, raw_results, overwrite=args.overwrite or offset > 0)
        if review_prompt_builder is not None:
            drafts = generated
            original_instruction = generator.developer_instruction
            generator.developer_instruction = review_developer_instruction or original_instruction
            try:
                generated = generator.generate([
                    review_prompt_builder(record, text) for record, text in zip(batch, generated)
                ])
            finally:
                generator.developer_instruction = original_instruction
            if len(generated) != len(batch):
                raise RuntimeError("evidence revision returned an incorrect record count")
            if review_raw_output_path is not None:
                review_results.extend({"id": record_id(record), "response": text}
                                      for record, text in zip(batch, generated))
                dump_json(review_raw_output_path, review_results,
                          overwrite=args.overwrite or offset > 0)
            if review_output_transform is not None:
                generated = [review_output_transform(record, draft, response)
                             for record, draft, response in zip(batch, drafts, generated)]
        if output_transform is not None:
            generated = [output_transform(text) for text in generated]
        if record_output_transform is not None:
            generated = [record_output_transform(record, text)
                         for record, text in zip(batch, generated)]
        results.extend(
            {**(output_fields or {}), "id": record_id(record), "reasoning": text}
            for record, text in zip(batch, generated)
        )
        print(f"generated {min(offset + len(batch), len(selected))}/{len(selected)}", flush=True)

    dump_json(args.output, results, overwrite=args.overwrite)
