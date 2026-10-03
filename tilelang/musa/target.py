from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def normalize_musa_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0].lower()
    if normalized.startswith("mp_"):
        return normalized
    if normalized.startswith("mp") and normalized[2:].isdigit():
        return f"mp_{normalized[2:]}"
    if normalized.isdigit():
        return f"mp_{normalized}"
    return None


def musa_arch_to_compute_version(arch: str | None) -> tuple[int, int] | None:
    """Convert an MP architecture such as ``mp_31`` to ``(3, 1)``."""
    normalized = normalize_musa_arch(arch)
    if normalized is None:
        return None
    arch_value = normalized[3:]
    if not arch_value.isdigit():
        return None
    return divmod(int(arch_value), 10)


def musa_arch_to_warp_size(arch: str | None) -> int | None:
    """Return the logical MUSA warp size encoded by an MP architecture.

    MP21/MP22 use 128-lane warps in the MUSA templates.  The MP31 target uses
    the 32-lane shuffle ABI.  Unknown architectures stay unresolved so target
    normalization cannot invent a lowering contract.
    """
    compute_version = musa_arch_to_compute_version(arch)
    if compute_version is None:
        return None
    major, minor = compute_version
    if major == 2 and minor in (1, 2):
        return 128
    if (major, minor) == (3, 1):
        return 32
    return None


def _detect_torch_musa_arch() -> str | None:
    """Return the architecture of the current torch MUSA device."""
    try:
        import torch

        if not hasattr(torch, "musa") or not torch.musa.is_available():
            return None
        device = torch.musa.current_device()
        properties = torch.musa.get_device_properties(device)
        return normalize_musa_arch(f"mp_{int(properties.major)}{int(properties.minor)}")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _detect_musa_target() -> Target | str | None:
    """Detect an available MUSA target and attach its MP architecture."""
    try:
        from tvm import musa as tvm_musa

        if not tvm_musa().exist:
            return None
    except (AttributeError, RuntimeError, ValueError):
        return None

    arch = _detect_torch_musa_arch()
    if arch is None:
        return "musa"
    return Target({"kind": "musa", "arch": arch})


def target_get_arch(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    return normalize_musa_arch(target.attrs.get("arch"))


def with_musa_target_attrs(target: Target) -> Target:
    if target.kind.name != "musa":
        return target
    arch = target_get_arch(target)
    if arch is None:
        return target
    target_dict = dict(target.export())
    target_dict["arch"] = arch
    warp_size = musa_arch_to_warp_size(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    return Target(target_dict)


def normalize_musa_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        parsed_target = target
    elif isinstance(target, dict):
        if target.get("kind") != "musa":
            return None
        try:
            parsed_target = Target(target)
        except Exception:
            return None
    else:
        return None

    if parsed_target.kind.name != "musa":
        return None
    return with_musa_target_attrs(parsed_target)


def target_is_musa(target: Target) -> bool:
    return _target_ffi_api().TargetIsMUSA(target)


def target_is_mp31(target: Target) -> bool:
    return _target_ffi_api().TargetIsMP31(target)


register_target_detector("musa", _detect_musa_target, override=True)
register_target_normalizer("musa", normalize_musa_target, override=True)
