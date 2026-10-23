from __future__ import annotations

import os
import time
from typing import List, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel, BertTokenizer

DEFAULT_BERT_PATH = os.environ.get(
    "QWENVL_CHEXBERT_BERT_PATH",
    "bert-base-uncased",
)
DEFAULT_CKPT_PATH = os.environ.get(
    "QWENVL_CHEXBERT_CKPT_PATH",
    "",
)
DEFAULT_REASONING_CKPT_PATH = os.environ.get(
    "QWENVL_REASONING_CHEXBERT_CKPT_PATH",
    "",
)
DEFAULT_REASONING_MAX_LENGTH = int(os.environ.get("QWENVL_REASONING_CHEXBERT_MAX_LENGTH", "2048"))


def _generate_attention_masks(batch_ids: torch.LongTensor) -> torch.FloatTensor:
    lengths = (batch_ids != 0).sum(dim=1)
    max_len = batch_ids.size(1)
    idxs = torch.arange(max_len, device=batch_ids.device).unsqueeze(0)
    return (idxs < lengths.unsqueeze(1)).float()


class BertLabeler(nn.Module):
    """BERT backbone plus CheXbert classification heads."""

    def __init__(
        self,
        *,
        device: Union[str, torch.device],
        bert_path: str = DEFAULT_BERT_PATH,
        checkpoint_path: str = DEFAULT_CKPT_PATH,
    ):
        super().__init__()

        if not checkpoint_path:
            raise ValueError("QWENVL_CHEXBERT_CKPT_PATH must point to a CheXbert checkpoint")

        self.device = torch.device(device) if isinstance(device, str) else device
        load_start = time.time()
        print(f"[reward][CheXbert] loading config from {bert_path}", flush=True)
        config = AutoConfig.from_pretrained(bert_path)
        print(
            f"[reward][CheXbert] config ready in {time.time() - load_start:.3f}s; loading BertModel from checkpoint path",
            flush=True,
        )
        self.bert = BertModel(config)

        hidden_size = self.bert.config.hidden_size
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2))
        self.dropout = nn.Dropout(0.1)

        ckpt_start = time.time()
        print(f"[reward][CheXbert] loading classifier checkpoint from {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=True)
        print(
            f"[reward][CheXbert] checkpoint loaded in {time.time() - ckpt_start:.3f}s; moving model to {self.device}",
            flush=True,
        )

        self.to(self.device)
        print(
            f"[reward][CheXbert] model moved to {self.device} in {time.time() - ckpt_start:.3f}s",
            flush=True,
        )
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def cls_logits(self, input_ids: torch.LongTensor) -> List[torch.Tensor]:
        attention_mask = _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return [head(cls_repr) for head in self.linear_heads]


class ChunkedBertLabeler(nn.Module):
    """CheXbert classifier with chunked long-text inference."""

    def __init__(
        self,
        bert_path: str,
        checkpoint_path: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not checkpoint_path:
            raise ValueError(
                "QWENVL_REASONING_CHEXBERT_CKPT_PATH must point to the reasoning classifier checkpoint"
            )
        build_start = time.time()
        print(f"[reward][ReasoningCheXbert] loading config from {bert_path}", flush=True)
        config = AutoConfig.from_pretrained(bert_path)
        print(
            f"[reward][ReasoningCheXbert] config ready in {time.time() - build_start:.3f}s; "
            "constructing BertModel from config",
            flush=True,
        )
        print("[reward][ReasoningCheXbert] BertModel(config) begin", flush=True)
        self.bert = BertModel(config)
        print(
            f"[reward][ReasoningCheXbert] BertModel(config) end in {time.time() - build_start:.3f}s",
            flush=True,
        )
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2))

        ckpt_start = time.time()
        print(f"[reward][ReasoningCheXbert] loading reasoning checkpoint from {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=True)
        print(
            f"[reward][ReasoningCheXbert] reasoning checkpoint loaded in {time.time() - ckpt_start:.3f}s",
            flush=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        if input_ids.dim() == 2:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
            return [head(cls_repr) for head in self.linear_heads]

        batch_size, num_chunks, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_chunks, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_chunks, seq_len)
        hidden_size = self.bert.config.hidden_size

        if chunk_mask is None:
            chunk_presence = (attention_mask.sum(dim=-1) > 0).float()
        else:
            chunk_presence = chunk_mask.float()

        flat_chunk_presence = chunk_presence.view(batch_size * num_chunks) > 0
        chunk_cls = torch.zeros(
            batch_size * num_chunks,
            hidden_size,
            device=input_ids.device,
            dtype=self.bert.embeddings.word_embeddings.weight.dtype,
        )

        if flat_chunk_presence.any():
            valid_input_ids = flat_input_ids[flat_chunk_presence]
            valid_attention_mask = flat_attention_mask[flat_chunk_presence]
            outputs = self.bert(input_ids=valid_input_ids, attention_mask=valid_attention_mask)
            valid_cls = self.dropout(outputs.last_hidden_state[:, 0])
            chunk_cls[flat_chunk_presence] = valid_cls

        chunk_cls = chunk_cls.view(batch_size, num_chunks, hidden_size)
        denom = chunk_presence.sum(dim=1, keepdim=True).clamp_min(1.0)
        cls_repr = (chunk_cls * chunk_presence.unsqueeze(-1)).sum(dim=1) / denom
        return [head(cls_repr) for head in self.linear_heads]


class CheXbertLabeler:
    """Local CheXbert wrapper for report-to-label conversion."""

    def __init__(
        self,
        *,
        device: Union[str, torch.device] | None = None,
        bert_path: str = DEFAULT_BERT_PATH,
        checkpoint_path: str = DEFAULT_CKPT_PATH,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        tok_start = time.time()
        print(f"[reward][CheXbert] loading tokenizer from {bert_path}", flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        print(
            f"[reward][CheXbert] tokenizer ready in {time.time() - tok_start:.3f}s",
            flush=True,
        )
        self.model = BertLabeler(
            device=self.device,
            bert_path=bert_path,
            checkpoint_path=checkpoint_path,
        )

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        input_ids = self.tokenizer(
            report,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).input_ids.to(self.device)
        preds = [head.argmax(dim=1).item() for head in self.model.cls_logits(input_ids)]

        binary: List[int] = []
        if mode == "rrg":
            for label in preds:
                binary.append(1 if label in {1, 3} else 0)
            return binary
        if mode == "classification":
            for label in preds:
                if label == 1:
                    binary.append(1)
                elif label == 3:
                    binary.append(-1)
                else:
                    binary.append(0)
            return binary
        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def get_labels(
        self,
        reports: List[str],
        mode: str = "rrg",
        batch_size: int = 16,
    ) -> List[List[int]]:
        if not reports:
            return []

        outputs: List[List[int]] = []
        for start in range(0, len(reports), max(1, int(batch_size))):
            batch_reports = reports[start : start + max(1, int(batch_size))]
            input_ids = self.tokenizer(
                batch_reports,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).input_ids.to(self.device)
            logits = self.model.cls_logits(input_ids)
            pred_rows = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().tolist()
            for preds in pred_rows:
                binary: List[int] = []
                if mode == "rrg":
                    for label in preds:
                        binary.append(1 if label in {1, 3} else 0)
                elif mode == "classification":
                    for label in preds:
                        if label == 1:
                            binary.append(1)
                        elif label == 3:
                            binary.append(-1)
                        else:
                            binary.append(0)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                outputs.append(binary)
        return outputs


class ReasoningCheXbertLabeler:
    """CheXbert judge trained on reasoning text with chunked long-context inference."""

    def __init__(
        self,
        *,
        device: Union[str, torch.device] | None = None,
        bert_path: str = DEFAULT_BERT_PATH,
        checkpoint_path: str = DEFAULT_REASONING_CKPT_PATH,
        max_length: int = DEFAULT_REASONING_MAX_LENGTH,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        tok_start = time.time()
        print(f"[reward][ReasoningCheXbert] loading tokenizer from {bert_path}", flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        print(
            f"[reward][ReasoningCheXbert] tokenizer ready in {time.time() - tok_start:.3f}s",
            flush=True,
        )
        self.max_length = max_length
        cfg_start = time.time()
        print(f"[reward][ReasoningCheXbert] loading max position config from {bert_path}", flush=True)
        config = AutoConfig.from_pretrained(bert_path)
        self.model_chunk_size = int(getattr(config, "max_position_embeddings", 512))
        print(
            f"[reward][ReasoningCheXbert] config ready in {time.time() - cfg_start:.3f}s; "
            f"model_chunk_size={self.model_chunk_size}",
            flush=True,
        )
        model_start = time.time()
        print("[reward][ReasoningCheXbert] ChunkedBertLabeler construction begin", flush=True)
        self.model = ChunkedBertLabeler(
            bert_path=bert_path,
            checkpoint_path=checkpoint_path,
        ).to(self.device).eval()
        print(
            f"[reward][ReasoningCheXbert] ChunkedBertLabeler construction end; full model moved to {self.device} in {time.time() - model_start:.3f}s",
            flush=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def _encode_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        token_ids = encoding["input_ids"].squeeze(0)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer must define cls_token_id and sep_token_id.")

        chunk_token_limit = max(self.model_chunk_size - 2, 1)
        token_chunks = [token_ids] if token_ids.numel() == 0 else list(token_ids.split(chunk_token_limit))

        chunk_input_ids = []
        chunk_attention_masks = []
        for chunk in token_chunks:
            ids = torch.cat(
                [
                    torch.tensor([cls_id], dtype=torch.long),
                    chunk.to(torch.long),
                    torch.tensor([sep_id], dtype=torch.long),
                ]
            )
            mask = torch.ones_like(ids, dtype=torch.long)
            chunk_input_ids.append(ids)
            chunk_attention_masks.append(mask)

        max_seq_len = max(ids.size(0) for ids in chunk_input_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(chunk_input_ids, chunk_attention_masks):
            pad_len = max_seq_len - ids.size(0)
            padded_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)], dim=0))
            padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)], dim=0))

        input_ids = torch.stack(padded_ids, dim=0).unsqueeze(0)
        attention_mask = torch.stack(padded_masks, dim=0).unsqueeze(0).float()
        chunk_mask = torch.ones((1, len(chunk_input_ids)), dtype=torch.float)
        return input_ids, attention_mask, chunk_mask

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        input_ids, attention_mask, chunk_mask = self._encode_text(report or "")
        logits = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            chunk_mask=chunk_mask.to(self.device),
        )
        preds = [head.argmax(dim=1).item() for head in logits]

        binary: List[int] = []
        if mode == "rrg":
            for label in preds:
                if label == 1:
                    binary.append(1)
                elif label == 3:
                    binary.append(1)
                else:
                    binary.append(0)
            return binary
        if mode == "classification":
            for label in preds:
                if label == 1:
                    binary.append(1)
                elif label == 3:
                    binary.append(-1)
                else:
                    binary.append(0)
            return binary
        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def get_labels(
        self,
        reports: List[str],
        mode: str = "rrg",
        batch_size: int = 4,
    ) -> List[List[int]]:
        if not reports:
            return []

        outputs: List[List[int]] = []
        micro_batch_size = max(1, int(batch_size))
        for start in range(0, len(reports), micro_batch_size):
            batch_reports = reports[start : start + micro_batch_size]
            encoded_batch = [self._encode_text(report or "") for report in batch_reports]

            max_chunks = max(int(input_ids.shape[1]) for input_ids, _, _ in encoded_batch)
            max_seq_len = max(int(input_ids.shape[2]) for input_ids, _, _ in encoded_batch)

            batch_input_ids = torch.zeros(
                (len(encoded_batch), max_chunks, max_seq_len),
                dtype=torch.long,
                device=self.device,
            )
            batch_attention_mask = torch.zeros(
                (len(encoded_batch), max_chunks, max_seq_len),
                dtype=torch.float,
                device=self.device,
            )
            batch_chunk_mask = torch.zeros(
                (len(encoded_batch), max_chunks),
                dtype=torch.float,
                device=self.device,
            )

            for row_idx, (input_ids, attention_mask, chunk_mask) in enumerate(encoded_batch):
                num_chunks = int(input_ids.shape[1])
                seq_len = int(input_ids.shape[2])
                batch_input_ids[row_idx, :num_chunks, :seq_len] = input_ids.squeeze(0).to(self.device)
                batch_attention_mask[row_idx, :num_chunks, :seq_len] = attention_mask.squeeze(0).to(self.device)
                batch_chunk_mask[row_idx, :num_chunks] = chunk_mask.squeeze(0).to(self.device)

            logits = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                chunk_mask=batch_chunk_mask,
            )
            pred_rows = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().tolist()
            for preds in pred_rows:
                binary: List[int] = []
                if mode == "rrg":
                    for label in preds:
                        if label in {1, 3}:
                            binary.append(1)
                        else:
                            binary.append(0)
                elif mode == "classification":
                    for label in preds:
                        if label == 1:
                            binary.append(1)
                        elif label == 3:
                            binary.append(-1)
                        else:
                            binary.append(0)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                outputs.append(binary)
        return outputs
