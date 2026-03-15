"""
Runtime backend selection for RL SDK clients.

MetaClaw can talk to either:
  - ``tinker`` directly
  - ``mint`` via the MindLab compatibility package
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .config import MetaClawConfig

_VALID_BACKENDS = {"auto", "tinker", "mint"}
_BACKEND_LABELS = {"tinker": "Tinker", "mint": "MinT"}
MINT_COMPAT_PACKAGE = "mindlab-toolkit"
MINT_TINKER_VERSION = "0.6.0"
MINT_INSTALL_COMMAND = 'pip install -e ".[mint]"'


@dataclass(frozen=True)
class SDKBackend:
    key: str
    label: str
    import_name: str
    module: Any
    api_key: str
    base_url: str

    @property
    def banner(self) -> str:
        return f"{self.label} cloud RL"


def _first_non_empty(*values: str) -> str:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _first_env(*names: str) -> str:
    return _first_non_empty(*(os.environ.get(name, "") for name in names))


def _normalize_backend_name(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid RL backend {value!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )
    return normalized


def configured_backend_name(config: "MetaClawConfig") -> str:
    return _normalize_backend_name(getattr(config, "backend", "auto"))


def configured_api_key(config: "MetaClawConfig") -> str:
    return _first_non_empty(
        getattr(config, "api_key", ""),
        getattr(config, "tinker_api_key", ""),
    )


def configured_base_url(config: "MetaClawConfig") -> str:
    return _first_non_empty(
        getattr(config, "base_url", ""),
        getattr(config, "tinker_base_url", ""),
    )


def _backend_env_order(kind: str, backend_key: str) -> tuple[str, str]:
    if kind not in {"api_key", "base_url"}:
        raise ValueError(f"Unknown backend env kind: {kind}")
    suffix = "API_KEY" if kind == "api_key" else "BASE_URL"
    if backend_key == "mint":
        return (f"MINT_{suffix}", f"TINKER_{suffix}")
    return (f"TINKER_{suffix}", f"MINT_{suffix}")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _looks_like_mint_base_url(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    return "mint" in urlparse(candidate).netloc.lower()


def looks_like_mint_configuration(api_key: str, base_url: str) -> bool:
    del api_key
    # API key formats are not stable enough to use as an inference signal.
    return _looks_like_mint_base_url(base_url)


def _has_mint_signal(config: "MetaClawConfig") -> bool:
    if _first_env("MINT_API_KEY", "MINT_BASE_URL"):
        return True

    url_candidates = (
        configured_base_url(config),
        os.environ.get("TINKER_BASE_URL", ""),
    )
    return any(looks_like_mint_configuration("", value) for value in url_candidates)


def _mint_install_guidance() -> str:
    return (
        f"Install MinT support with: {MINT_INSTALL_COMMAND}. "
        f"That extra installs '{MINT_COMPAT_PACKAGE}' and the compatible "
        f"tinker=={MINT_TINKER_VERSION}."
    )


def _mint_missing_support_error(*, auto_detected: bool) -> RuntimeError:
    if auto_detected:
        prefix = "Detected MinT configuration via MINT_* env vars or base URL, but MinT support is not installed."
    else:
        prefix = "rl.backend=mint requires MinT support in this environment."
    return RuntimeError(f"{prefix} {_mint_install_guidance()}")


def _mint_import_error(exc: Exception, *, auto_detected: bool) -> RuntimeError:
    if auto_detected:
        prefix = "Detected MinT configuration via MINT_* env vars or base URL, but MinT could not be imported cleanly."
    else:
        prefix = "rl.backend=mint selected, but MinT could not be imported cleanly."
    return RuntimeError(
        f"{prefix} {_mint_install_guidance()} "
        f"Original import error ({type(exc).__name__}): {exc}"
    )


def infer_backend_key(config: "MetaClawConfig") -> str:
    configured = configured_backend_name(config)
    if configured in {"tinker", "mint"}:
        return configured
    if _has_mint_signal(config) and _module_available("mint"):
        return "mint"
    return "tinker"


def resolve_api_key(config: "MetaClawConfig", backend_key: str | None = None) -> str:
    configured = configured_api_key(config)
    if configured:
        return configured
    key = backend_key or infer_backend_key(config)
    return _first_env(*_backend_env_order("api_key", key))


def resolve_base_url(config: "MetaClawConfig", backend_key: str | None = None) -> str:
    configured = configured_base_url(config)
    if configured:
        return configured
    key = backend_key or infer_backend_key(config)
    return _first_env(*_backend_env_order("base_url", key))


def _import_backend_module(backend_key: str, configured_backend: str, *, auto_detected: bool):
    if backend_key != "mint":
        return importlib.import_module(backend_key)

    if not _module_available("mint"):
        raise _mint_missing_support_error(auto_detected=auto_detected)

    try:
        return importlib.import_module("mint")
    except Exception as exc:
        raise _mint_import_error(exc, auto_detected=auto_detected) from exc


def resolve_sdk_backend(config: "MetaClawConfig") -> SDKBackend:
    configured = configured_backend_name(config)
    auto_detected = configured == "auto" and _has_mint_signal(config)
    if auto_detected and not _module_available("mint"):
        raise _mint_missing_support_error(auto_detected=True)
    key = infer_backend_key(config)
    module = _import_backend_module(
        key,
        configured,
        auto_detected=auto_detected and key == "mint",
    )
    return SDKBackend(
        key=key,
        label=_BACKEND_LABELS[key],
        import_name=key,
        module=module,
        api_key=resolve_api_key(config, key),
        base_url=resolve_base_url(config, key),
    )
