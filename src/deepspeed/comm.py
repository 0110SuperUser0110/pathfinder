from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ReduceOp:
    SUM: str = "sum"


ReduceOp = _ReduceOp()


def is_initialized() -> bool:
    return False


def get_world_size() -> int:
    return 1


def all_reduce(tensor, op=None):
    return tensor


def broadcast(tensor, src=0):
    return tensor
