"""Single source of truth for the contract-v2 Qwen3 pin.

No module may load a language-model checkpoint without going through this
file.  There is no Qwen2.5 fallback, quantized substitute, or revision alias.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from suture.paths import REPO_ROOT


CONTRACT_VERSION = 2
ACTIVE_CONTRACT_VERSION = 2
CHECKPOINT_SELECTION_RULE = "max_gain_among_passing"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_TYPE = "qwen3"
MODEL_ARCHITECTURE = "Qwen3ForCausalLM"
N_LAYERS = 28
HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936
N_PARAMETERS = 2_031_739_904
N_PARAMETERS_UNIQUE = 1_720_574_976
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "eager"

# Historical Coder pin (contract v1). Recorded only so v2 can refuse it.
V1_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
V1_MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"

FORBIDDEN_MODEL_MARKERS = (
    "Qwen2.5",
    "Qwen2.5-Coder",
    "Qwen2-1.5B",
    "Qwen/Qwen2.5",
    "gptq",
    "awq",
    "gguf",
    "bitsandbytes",
    "int4",
    "int8",
    "4bit",
    "8bit",
)

LANGUAGES = ("es", "zh", "sw")
LANGUAGE_NAMES = {"es": "Spanish", "zh": "Chinese", "sw": "Swahili"}
EXPECTED_LID = {
    "es": "__label__es",
    "zh": "__label__zh",
    "sw": "__label__sw",
}

V2_RESULTS_ROOT = Path("results") / "v2" / "qwen3_1_7b"
V3_WITHDRAWN_ROOT = Path("results") / "v3" / "qwen3_1_7b"
V4_WITHDRAWN_ROOT = Path("results") / "v4" / "qwen3_1_7b"
RESULTS_ROOT = V2_RESULTS_ROOT
V1_CANONICAL_E1 = Path("results") / "tier_a" / "es" / "e1_canonical_seed0"
V2_CONTRACT_PATH = Path("configs") / "experimental_contract_v2.json"
V1_CONTRACT_PATH = Path("configs") / "experimental_contract_v1.json"

N_UTIL = 500
N_RISK = 500
N_CALIBRATION = 1000
N_HOST_TRAIN = 1000
N_MGSM_DEV = 64
N_MGSM_TEST = 186
N_MGSM_TOTAL = 250
N_EXPERT_VALIDATION = 64
N_READINESS_DONOR = 128
N_READINESS_HOST = 128
N_BELEBELE = 128

LORA_RANK = 8
TRAIN_STEPS = 1024
TRAIN_CONTINUATION_STEPS = 1536
TRAIN_BATCH_SIZE = 1
TRAIN_GRAD_ACCUM = 8
TRAIN_MAX_LENGTH = 128
TRAIN_LEARNING_RATE = 1e-5
CHECKPOINT_STEPS = (128, 256, 512, 768, 1024)
EXPERT_SEED = 0

MIN_DONOR_GAIN = 0.10
MIN_DONOR_EM = 0.20
MIN_HOST_NONEMPTY = 0.90
MIN_HOST_TARGET_LANGUAGE = 0.90
MIN_HOST_LOGPROB_GAIN = 0.05
MAX_BELEBELE_DROP = 0.05

ENABLE_THINKING = False
DECODE_TEMPERATURE = 0.7
DECODE_TOP_P = 0.8
DECODE_TOP_K = 20
MAX_NEW_TOKENS = 128
RANKING_DECODE_SEEDS = (0,)
STABILITY_DECODE_SEEDS = (0, 1, 2, 3, 4)
GENERATION_BATCH_SIZE = 1
SCORE_BATCH_SIZE = 1
TAU = 0.05
MIN_SPEARMAN = 0.5
MAX_REGRET_FRACTION = 0.5
MAX_DRIFT_RATE = 0.5
P1_MIN_SPEARMAN = 0.9
P3_MAX_RELATIVE_ERROR = 1e-4

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LID_MODEL_PATH = Path("models") / "lid.176.ftz"
LID_MODEL_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"

HOST_SOURCE_CANDIDATES: Dict[str, Sequence[Dict[str, str]]] = {
    "es": (
        {
            "dataset": "CohereLabsCommunity/multilingual-reward-bench",
            "config": "spa_Latn",
            "split": "test",
        },
        {
            "dataset": "somosnlp/somos-clean-alpaca-es",
            "config": "",
            "split": "train",
        },
        {
            "dataset": "bertin-project/alpaca-spanish",
            "config": "",
            "split": "train",
        },
    ),
    "zh": (
        {
            "dataset": "CohereLabsCommunity/multilingual-reward-bench",
            "config": "zho_Hans",
            "split": "test",
        },
        {
            "dataset": "shibing624/alpaca-zh",
            "config": "",
            "split": "train",
        },
    ),
    "sw": (
        {
            "dataset": "Mollel/alpaca-swahili",
            "config": "",
            "split": "train",
        },
        {
            "dataset": "iamshnoo/alpaca-cleaned-swahili",
            "config": "",
            "split": "train",
        },
    ),
}


class ConfigError(RuntimeError):
    """Raised when the Qwen3 pin, contract, or output path is violated."""


def repo_root() -> Path:
    return REPO_ROOT


def snapshot_dir(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "models" / f"Qwen3-1.7B-{MODEL_REVISION[:8]}"


def results_root(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / RESULTS_ROOT


def v2_contract_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / V2_CONTRACT_PATH


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_mgsm_indices(
    *,
    n_total: int = N_MGSM_TOTAL,
    n_dev: int = N_MGSM_DEV,
) -> Dict[str, tuple[int, ...]]:
    """Return answer-independent 64/186 MGSM index partitions.

    Item identity is the integer index in ``juletxara/mgsm:{lang}:test``.
    The same indices are used for every language.
    """

    if n_dev + (n_total - n_dev) != n_total:
        raise ConfigError("MGSM split sizes are inconsistent")
    ranked = sorted(
        range(n_total),
        key=lambda index: sha256_bytes(f"mgsm:item:{index}".encode("utf-8")),
    )
    return {
        "dev": tuple(sorted(ranked[:n_dev])),
        "test": tuple(sorted(ranked[n_dev:])),
    }


def refuse_forbidden_model(model_id: str, revision: str | None = None) -> None:
    blob = f"{model_id} {revision or ''}".lower()
    if model_id != MODEL_ID:
        raise ConfigError(
            f"refusing model {model_id!r}; contract v2 requires exactly {MODEL_ID}"
        )
    if revision is not None and revision != MODEL_REVISION:
        raise ConfigError(
            f"refusing revision {revision!r}; contract v2 requires {MODEL_REVISION}"
        )
    for marker in FORBIDDEN_MODEL_MARKERS:
        if marker.lower() in blob and marker.lower() not in MODEL_ID.lower():
            raise ConfigError(f"refusing forbidden model marker {marker!r} in {model_id}")


def refuse_v1_write(path: Path, *, root: Path | None = None) -> None:
    resolved = path.resolve()
    canonical = ((root or repo_root()) / V1_CANONICAL_E1).resolve()
    if resolved == canonical or canonical in resolved.parents:
        raise ConfigError(f"refusing write into frozen v1 canonical E1: {resolved}")
    frozen_root = ((root or repo_root()) / "results" / "tier_a").resolve()
    active_root = results_root(root).resolve()
    if frozen_root in resolved.parents or resolved == frozen_root:
        if active_root not in resolved.parents and resolved != active_root:
            raise ConfigError(f"new artifacts must not be written under {frozen_root}: {resolved}")


def refuse_legacy_write(path: Path, *, root: Path | None = None) -> None:
    resolved = path.resolve()
    base = root or repo_root()
    for rel, label in (
        (V3_WITHDRAWN_ROOT, "results/v3"),
        (V4_WITHDRAWN_ROOT, "results/v4"),
    ):
        withdrawn = (base / rel).resolve()
        if resolved == withdrawn or withdrawn in resolved.parents:
            raise ConfigError(f"refusing write into withdrawn {label} path: {resolved}")


def refuse_v4_write(path: Path, *, root: Path | None = None) -> None:
    refuse_legacy_write(path, root=root)


def refuse_frozen_write(path: Path, *, root: Path | None = None) -> None:
    refuse_v1_write(path, root=root)
    refuse_legacy_write(path, root=root)


def load_v2_contract(root: Path | None = None) -> Dict[str, Any]:
    path = v2_contract_path(root)
    if not path.is_file():
        raise ConfigError(f"v2 contract is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("contract_version", -1)) != CONTRACT_VERSION:
        raise ConfigError("v2 contract_version must be 2")
    model = payload.get("base_model", {})
    refuse_forbidden_model(str(model.get("id")), str(model.get("revision")))
    return payload


def apply_qwen3_chat(
    tokenizer: Any,
    user_text: str,
    *,
    system: str | None = None,
    enable_thinking: bool = ENABLE_THINKING,
) -> str:
    """Render a Qwen3 chat prompt.  Thinking mode is frozen off for v2."""

    if not getattr(tokenizer, "chat_template", None):
        return user_text
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=bool(enable_thinking),
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def assert_adapter_matches_pin(adapter_dir: Path) -> Dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        found = sorted(adapter_dir.rglob("adapter_config.json"))
        if not found:
            raise ConfigError(f"adapter_config.json not found under {adapter_dir}")
        config_path = found[0]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base = str(payload.get("base_model_name_or_path") or "")
    refuse_forbidden_model(base, MODEL_REVISION if base == MODEL_ID else None)
    if base != MODEL_ID:
        raise ConfigError(
            f"adapter at {adapter_dir} is bound to {base!r}, not {MODEL_ID}"
        )
    return payload


def model_record() -> Dict[str, Any]:
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_type": MODEL_TYPE,
        "architecture": MODEL_ARCHITECTURE,
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "n_parameters": N_PARAMETERS,
        "dtype": TORCH_DTYPE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "fallback": None,
        "contract_version": CONTRACT_VERSION,
    }


def load_kwargs(*, local_files_only: bool) -> Dict[str, Any]:
    refuse_forbidden_model(MODEL_ID, MODEL_REVISION)
    return {
        "revision": MODEL_REVISION,
        "local_files_only": bool(local_files_only),
        "dtype": __import__("torch").bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
