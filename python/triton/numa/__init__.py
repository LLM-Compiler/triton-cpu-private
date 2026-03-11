"""
NUMA extension for Triton-CPU.

Provides NUMA-aware distributed algorithms and allocation utilities
for multi-socket CPU systems.
"""

from . import alloc_utils
from . import distributed
from . import distributed_gemm

__all__ = ["alloc_utils", "distributed", "distributed_gemm"]
