"""Cache mean-pooled embeddings from frozen protein or RNA encoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol, Sequence

import torch
import yaml

from .proteingym import _resolve_device, discover_assays


LOGGER = logging.getLogger("lyra.extract_embeddings")


@dataclass(frozen=True)
class EmbeddingConfig:
    input_path: Path
    output_dir: Path
    checkpoint: str
    backend: str = "huggingface"
    sequence_column: str = "mutated_sequence"
    pattern: str = "*.csv"
    batch_size: int = 8
    max_residues: int = 1022
    device: str = "auto"
    dtype: str = "bfloat16"
    resume: bool = True
    layer_name: str | None = None
    trust_remote_code: bool = False


class Encoder(Protocol):
    checkpoint: str

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        """Return one float32 feature vector per sequence."""


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"dtype must be one of: {sorted(mapping)}") from error


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _sequence_sha256(sequences: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        encoded = sequence.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


class HuggingFaceEncoder:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
        except ImportError as error:
            raise ImportError(
                "Hugging Face embedding extraction requires transformers"
            ) from error
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, trust_remote_code=trust_remote_code
            )
        except ValueError as error:
            # biohub/ESMC-600M currently names ESMCTokenizer in its tokenizer
            # metadata without publishing an AutoTokenizer mapping. Its
            # tokenizer.json is nevertheless a complete Hugging Face fast
            # tokenizer, so load that representation directly. Keep this
            # fallback intentionally narrow so unrelated configuration errors
            # remain visible.
            if "Tokenizer class ESMCTokenizer does not exist" not in str(error):
                raise
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        if getattr(getattr(self.model, "config", None), "model_type", None) == "jamba":
            disabled = 0
            for module in self.model.modules():
                if hasattr(module, "use_fast_kernels"):
                    module.use_fast_kernels = False
                    disabled += 1
            if disabled:
                LOGGER.warning(
                    "Jamba fast Mamba kernels are optional; using the portable "
                    "PyTorch recurrence for %d mixer module(s)",
                    disabled,
                )
        self.model.requires_grad_(False).eval().to(device)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(sequences),
            add_special_tokens=True,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = batch.pop("special_tokens_mask").bool()
        inputs = {key: value.to(self.device) for key, value in batch.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype != torch.float32,
        ):
            output = self.model(**inputs)
        hidden = output.last_hidden_state
        mask = inputs["attention_mask"].bool() & ~special.to(self.device)
        return _masked_mean(hidden, mask).float().cpu()


class E1Encoder:
    def __init__(
        self, checkpoint: str, device: torch.device, dtype: torch.dtype
    ) -> None:
        try:
            from E1.batch_preparer import E1BatchPreparer
            from E1.modeling import E1ForMaskedLM
        except ImportError as error:
            raise ImportError(
                "Profluent E1 extraction requires the official Profluent-AI/E1 package"
            ) from error
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        self.preparer = E1BatchPreparer()
        self.model = E1ForMaskedLM.from_pretrained(checkpoint)
        self.model.requires_grad_(False).eval().to(device)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        batch = self.preparer.get_batch_kwargs(list(sequences), device=str(self.device))
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype != torch.float32,
        ):
            output = self.model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        residue_mask = ~self.preparer.get_boundary_token_mask(batch["input_ids"])
        return _masked_mean(output.embeddings, residue_mask).float().cpu()


class RNAFMEncoder:
    """Frozen RNA-FM layer-12 residue representations."""

    def __init__(self, checkpoint: str, device: torch.device) -> None:
        try:
            import fm
        except ImportError as error:
            raise ImportError(
                "RNA-FM extraction requires the official ml4bio/RNA-FM package"
            ) from error
        if checkpoint not in {"rna_fm_t12", "RNA-FM", "RNA_central"}:
            raise ValueError(
                "RNA-FM checkpoint must be one of: rna_fm_t12, RNA-FM, RNA_central"
            )
        self.checkpoint = checkpoint
        self.device = device
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.requires_grad_(False).eval().to(device)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        normalized = [sequence.replace("T", "U") for sequence in sequences]
        data = [(str(index), sequence) for index, sequence in enumerate(normalized)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        output = self.model(
            tokens,
            repr_layers=[12],
            need_head_weights=False,
            return_contacts=False,
        )
        hidden = output["representations"][12]
        # BatchConverter adds BOS and EOS. Slice by the original residue lengths so
        # neither boundary nor padding tokens enter the sequence representation.
        return torch.stack(
            [hidden[index, 1 : 1 + len(sequence)].mean(dim=0) for index, sequence in enumerate(normalized)]
        ).float().cpu()


class EvoEncoder:
    """Frozen Evo 1/1.5 representations captured before the language-model head."""

    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        *,
        layer_name: str | None = None,
    ) -> None:
        try:
            import huggingface_hub
            from evo import Evo
        except ImportError as error:
            raise ImportError(
                "Evo extraction requires the official evo-design/evo package"
            ) from error
        self.checkpoint = checkpoint
        self.device = device
        self.layer_name = layer_name or "norm"
        # Evo's current loader reads safetensors, but its unfiltered
        # snapshot_download also fetches duplicate PyTorch .bin/.pt weights.
        # Restrict the snapshot to the metadata and weight format it consumes.
        original_snapshot_download = huggingface_hub.snapshot_download

        def safetensors_snapshot(*args, **kwargs):
            kwargs.setdefault("allow_patterns", ["*.json", "*.safetensors"])
            return original_snapshot_download(*args, **kwargs)

        huggingface_hub.snapshot_download = safetensors_snapshot
        try:
            wrapper = Evo(checkpoint, device=str(device))
        finally:
            huggingface_hub.snapshot_download = original_snapshot_download
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        normalized = [sequence.replace("U", "T") for sequence in sequences]
        tokenized = [self.tokenizer.tokenize(sequence) for sequence in normalized]
        max_length = max(map(len, tokenized))
        tokens = torch.full(
            (len(tokenized), max_length),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros(tokens.shape, dtype=torch.bool, device=self.device)
        for index, values in enumerate(tokenized):
            tokens[index, : len(values)] = torch.tensor(values, device=self.device)
            mask[index, : len(values)] = True
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _inputs, output) -> None:
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = self.model.get_submodule(self.layer_name).register_forward_hook(capture)
        try:
            self.model(tokens, padding_mask=mask)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError(f"Evo layer hook {self.layer_name!r} produced no output")
        return _masked_mean(captured["hidden"], mask).float().cpu()


class Evo2Encoder:
    """Frozen Evo 2 intermediate-layer representations."""

    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        *,
        layer_name: str | None,
    ) -> None:
        try:
            from evo2 import Evo2
        except ImportError as error:
            raise ImportError(
                "Evo 2 extraction requires the official ArcInstitute/evo2 package"
            ) from error
        if layer_name is None:
            raise ValueError(
                "Evo 2 requires layer_name; use blocks.26 for the official 7B "
                "downstream representation example"
            )
        if device.type != "cuda":
            raise ValueError("Evo 2 requires a CUDA device")
        self.checkpoint = checkpoint
        self.device = device
        self.layer_name = layer_name
        self.wrapper = Evo2(checkpoint)
        self.wrapper.model.requires_grad_(False).eval()
        self.tokenizer = self.wrapper.tokenizer

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        # Vortex does not accept a padding mask through Evo2.forward, so group by
        # exact length. This preserves batching without allowing pad tokens to
        # change the causal representation.
        normalized = [sequence.replace("U", "T") for sequence in sequences]
        by_length: dict[int, list[int]] = {}
        for index, sequence in enumerate(normalized):
            by_length.setdefault(len(sequence), []).append(index)
        features: list[torch.Tensor | None] = [None] * len(normalized)
        for indices in by_length.values():
            tokens = torch.tensor(
                [self.tokenizer.tokenize(normalized[index]) for index in indices],
                dtype=torch.int,
                device=self.device,
            )
            _, embeddings = self.wrapper(
                tokens,
                return_embeddings=True,
                layer_names=[self.layer_name],
            )
            hidden = embeddings[self.layer_name]
            pooled = hidden.mean(dim=1).float().cpu()
            for index, feature in zip(indices, pooled, strict=True):
                features[index] = feature
        if any(feature is None for feature in features):
            raise RuntimeError("Evo 2 failed to produce every requested embedding")
        return torch.stack([feature for feature in features if feature is not None])


def _build_encoder(config: EmbeddingConfig, device: torch.device) -> Encoder:
    dtype = _torch_dtype(config.dtype)
    if config.backend == "huggingface":
        return HuggingFaceEncoder(
            config.checkpoint,
            device,
            dtype,
            trust_remote_code=config.trust_remote_code,
        )
    if config.backend == "e1":
        return E1Encoder(config.checkpoint, device, dtype)
    if config.backend == "rna_fm":
        return RNAFMEncoder(config.checkpoint, device)
    if config.backend == "evo":
        return EvoEncoder(
            config.checkpoint, device, layer_name=config.layer_name
        )
    if config.backend == "evo2":
        return Evo2Encoder(
            config.checkpoint, device, layer_name=config.layer_name
        )
    raise ValueError("backend must be one of: huggingface, e1, rna_fm, evo, evo2")


def _read_sequences(path: Path, sequence_column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if sequence_column not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: missing sequence column {sequence_column!r}")
        sequences = [row[sequence_column].strip().upper() for row in reader]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError(f"{path}: sequences must be non-empty")
    return sequences


def _encode_with_windows(
    encoder: Encoder,
    sequences: Sequence[str],
    *,
    batch_size: int,
    max_residues: int,
) -> torch.Tensor:
    """Mean-pool complete sequences, using non-overlapping context windows."""

    all_features: list[torch.Tensor] = []
    for batch_start in range(0, len(sequences), batch_size):
        sequence_batch = sequences[batch_start : batch_start + batch_size]
        chunks: list[str] = []
        owners: list[int] = []
        lengths: list[int] = []
        for owner, sequence in enumerate(sequence_batch):
            for start in range(0, len(sequence), max_residues):
                chunk = sequence[start : start + max_residues]
                chunks.append(chunk)
                owners.append(owner)
                lengths.append(len(chunk))

        encoded_chunks: list[torch.Tensor] = []
        for start in range(0, len(chunks), batch_size):
            encoded_chunks.append(
                _encode_with_oom_backoff(
                    encoder, chunks[start : start + batch_size]
                )
            )
        chunk_features = torch.cat(encoded_chunks)
        batch_features = torch.zeros(
            len(sequence_batch), chunk_features.size(1), dtype=torch.float32
        )
        total_lengths = torch.zeros(len(sequence_batch), dtype=torch.float32)
        for feature, owner, length in zip(
            chunk_features, owners, lengths, strict=True
        ):
            batch_features[owner] += feature * length
            total_lengths[owner] += length
        batch_features /= total_lengths.unsqueeze(-1)
        all_features.append(batch_features)
    return torch.cat(all_features)


def _encode_with_oom_backoff(
    encoder: Encoder, sequences: Sequence[str]
) -> torch.Tensor:
    """Encode a batch, recursively halving it after a CUDA capacity failure.

    Sequence datasets often mix many short variants with a few full-context
    records. Keeping the configured batch size for the common case is much
    faster than globally forcing batch size one, while recursive backoff makes
    those exceptional long records resumable.
    """

    try:
        return encoder.encode(sequences)
    except RuntimeError as error:
        # CUDA libraries do not consistently surface workspace allocation
        # failures as torch.OutOfMemoryError. Evo2's FFT convolution can emit
        # CUFFT_ALLOC_FAILED instead; cuBLAS uses an analogous status code.
        capacity_failure = isinstance(error, torch.OutOfMemoryError) or any(
            marker in str(error).upper()
            for marker in (
                "CUDA OUT OF MEMORY",
                "CUFFT_ALLOC_FAILED",
                "CUBLAS_STATUS_ALLOC_FAILED",
                "CANUSE32BITINDEXMATH",
            )
        )
        if not capacity_failure:
            raise
        if len(sequences) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(sequences) // 2
        LOGGER.warning(
            "CUDA capacity failure for embedding batch of %d; retrying as %d + %d",
            len(sequences),
            midpoint,
            len(sequences) - midpoint,
        )
        return torch.cat(
            (
                _encode_with_oom_backoff(encoder, sequences[:midpoint]),
                _encode_with_oom_backoff(encoder, sequences[midpoint:]),
            )
        )


def extract_embeddings(config: EmbeddingConfig) -> list[Path]:
    if config.batch_size < 1 or config.max_residues < 1:
        raise ValueError("batch_size and max_residues must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    encoder = _build_encoder(config, device)
    outputs: list[Path] = []
    for index, path in enumerate(
        discover_assays(config.input_path, config.pattern), start=1
    ):
        output_path = config.output_dir / f"{path.stem}.pt"
        sequences = _read_sequences(path, config.sequence_column)
        if output_path.exists() and config.resume:
            payload = torch.load(output_path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            expected = {
                "checkpoint": config.checkpoint,
                "backend": config.backend,
                "max_residues": config.max_residues,
                "n_samples": len(sequences),
                "sequence_sha256": _sequence_sha256(sequences),
                "layer_name": config.layer_name,
            }
            if all(metadata.get(key) == value for key, value in expected.items()):
                LOGGER.info("[%d] %s already cached; skipping", index, path.stem)
                outputs.append(output_path)
                continue
            raise ValueError(
                f"{output_path}: cached embedding metadata does not match the "
                "requested checkpoint, layer, windowing, or source sequences"
            )
        LOGGER.info("[%d] %s: embedding %d sequences", index, path.stem, len(sequences))
        features = _encode_with_windows(
            encoder,
            sequences,
            batch_size=config.batch_size,
            max_residues=config.max_residues,
        )
        torch.save(
            {
                "features": features.to(torch.float16),
                "metadata": {
                    "checkpoint": config.checkpoint,
                    "backend": config.backend,
                    "pooling": "complete_sequence_residue_mean",
                    "max_residues": config.max_residues,
                    "n_samples": len(sequences),
                    "hidden_dim": features.size(1),
                    "sequence_sha256": _sequence_sha256(sequences),
                    "layer_name": config.layer_name,
                },
            },
            output_path,
        )
        outputs.append(output_path)

    run_config = asdict(config)
    run_config.update(input_path=str(config.input_path), output_dir=str(config.output_dir))
    with (config.output_dir / "embedding_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    return outputs


def load_config(path: Path) -> EmbeddingConfig:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    allowed = {field.name for field in fields(EmbeddingConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown configuration keys: {unknown}")
    required = {"input_path", "output_dir", "checkpoint"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    base = path.resolve().parent
    for key in ("input_path", "output_dir"):
        value = Path(values[key])
        values[key] = value if value.is_absolute() else base / value
    return EmbeddingConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cache frozen protein embeddings")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_embeddings(load_config(args.config))


if __name__ == "__main__":
    main()
