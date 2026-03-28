# Nontemporal Instruction Generation — Repository Analysis

> **Scope**: This document analyzes all occurrences where nontemporal instructions are generated in the Triton-CPU codebase, and whether nontemporal instructions were emitted **before** the recent NUMA changes (commits starting from `2adfa59`).

---

## 1. Summary

| Question | Answer |
|----------|--------|
| **Were nontemporal instructions emitted before the NUMA changes?** | **No.** The CPU backend did not propagate `CacheModifier::CG` to nontemporal LLVM/vector ops before commit `2bd6814b`. |
| **Where are nontemporal instructions generated now?** | Two locations: `ConvertMemoryOps.cpp` (vector dialect) and `MemoryOpToLLVM.cpp` (LLVM dialect). |
| **What triggers nontemporal?** | `CacheModifier::CG` on `triton::LoadOp` or `triton::StoreOp`. Sources: NUMA ops (via ConvertNUMAOps) or explicit `tl.load(..., cache_modifier=".cg")` / `tl.store(..., cache_modifier=".cg")`. |

---

## 2. All Occurrences of Nontemporal Generation

### 2.1 ConvertMemoryOps.cpp (TTCIR → Vector Dialect)

**File**: `third_party/cpu/lib/TritonToTritonCPU/ConvertMemoryOps.cpp`

| Location | Op | Condition | Effect |
|----------|-----|-----------|--------|
| **Lines 253–256** | `vector::LoadOp` | `loadOp.getCache() == CacheModifier::CG` | Sets `nontemporal=true` on `vector::LoadOp` in `lowerToContiguousRowMajor` |
| **Lines 443–445** | `vector::StoreOp` | `storeOp.getCache() == CacheModifier::CG` | Sets `nontemporal=true` on `vector::StoreOp` in `lowerToContiguousRowMajor` |

**Code path**: `LoadOpConversion::matchAndRewrite` → `lowerToContiguousRowMajor` (when `isContiguousRowMajorAccess` and no mask) → `vector::LoadOp` / `vector::StoreOp` with `/*nontemporal=*/nt`.

**Downstream**: MLIR's `ConvertVectorToLLVM` pass (via `add_vector_to_llvmir`) lowers `vector::LoadOp`/`vector::StoreOp` to LLVM IR and propagates the nontemporal attribute to `!nontemporal` metadata on the resulting `load`/`store` instructions.

---

### 2.2 MemoryOpToLLVM.cpp (TTCIR → LLVM Dialect)

**File**: `third_party/cpu/lib/TritonCPUToLLVM/MemoryOpToLLVM.cpp`

| Location | Op | Condition | Effect |
|----------|-----|-----------|--------|
| **Lines 208–210** | `LLVM::LoadOp` | `op.getCache() == CacheModifier::CG` | Passes `nonTemporal=true` as 5th argument to `LLVM::LoadOp` |
| **Lines 223–226** | `LLVM::StoreOp` | `op.getCache() == CacheModifier::CG` | Passes `nonTemporal=true` as 5th argument to `LLVM::StoreOp` |

**Code path**: `LoadOpConversion` and `StoreOpConversion` match scalar `triton::LoadOp`/`triton::StoreOp` that reach the LLVM lowering stage (e.g., from `lowerToScalarLoads`/`lowerToScalarStores` when the contiguous/gather paths are not taken).

**Downstream**: LLVM backend emits `vmovntdqa` (loads) and `vmovntps`/`vmovntdq` (stores) on x86 when `!nontemporal` metadata is present.

---

### 2.3 Paths That Do NOT Generate Nontemporal

| Path | Reason |
|------|--------|
| `vector::MaskedLoadOp` / `vector::MaskedStoreOp` | These ops do not support a nontemporal attribute in the current MLIR version. |
| `vector::GatherOp` / `vector::ScatterOp` | No nontemporal parameter; cache modifier is not propagated. |
| `lowerToGather` / `lowerToScatter` | Creates `vector::GatherOp`/`vector::ScatterOp` without nontemporal. |
| `ConvertDotToFMA.cpp` | Creates `vector::LoadOp`/`vector::StoreOp` without nontemporal (internal buffers). |
| `ConvertUnsupportedOps.cpp` | `vector::MaskedLoadOp`/`vector::MaskedStoreOp` for BF16 conversion. |

---

## 3. Sources of CacheModifier::CG

| Source | File | How CG is set |
|--------|------|---------------|
| **NUMA ops** | `ConvertNUMAOps.cpp` | `analyzeReuseDistance()` returns `CG` for streaming patterns; `NumaStoreToStoreConversion` always uses `CG`. |
| **User explicit** | `semantic.py` | `tl.load(..., cache_modifier=".cg")` → `CacheModifier::CG` via `_str_to_load_cache_modifier`. |
| **User explicit** | `semantic.py` | `tl.store(..., cache_modifier=".cg")` → `CacheModifier::CG` via `_str_to_store_cache_modifier`. |
| **Default** | `TritonOps.td` | `LoadOp` and `StoreOp` default to `CacheModifier::NONE` when no modifier is specified. |

---

## 4. Were Nontemporal Instructions Emitted Before the NUMA Changes?

### 4.1 State Before Commit 2bd6814b

**MemoryOpToLLVM.cpp** (before 2bd6814b):

```cpp
// LoadOpConversion - NO nonTemporal
rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resTy, ptr, 0,
                                          op.getIsVolatile());

// StoreOpConversion - NO nonTemporal
rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, val, ptr);
```

`LLVM::LoadOp` was called with 4 arguments (no `nonTemporal`). `LLVM::StoreOp` was called with 3 arguments (no alignment, volatile, or nontemporal).

**ConvertMemoryOps.cpp** (before 63d27de):

```cpp
// lowerToContiguousRowMajor - NO nontemporal
vec = rewriter.create<vector::LoadOp>(loc, subVecTy, memRef, zeroIdx);
// ...
rewriter.create<vector::StoreOp>(loc, val, memRef, zeroIdx);
```

`vector::LoadOp` and `vector::StoreOp` were created without the `nontemporal` parameter.

### 4.2 Conclusion

**Nontemporal instructions were NOT emitted before the NUMA-related changes.**

| Component | Before | After |
|-----------|--------|-------|
| **MemoryOpToLLVM** | Ignored `op.getCache()`; no `nonTemporal` passed to `LLVM::LoadOp`/`LLVM::StoreOp` | `op.getCache() == CG` → `nonTemporal=true` |
| **ConvertMemoryOps** | Did not pass `nontemporal` to `vector::LoadOp`/`vector::StoreOp` | `loadOp.getCache() == CG` → `nontemporal=true` |

Even if a user wrote `tl.load(ptr, cache_modifier=".cg")` before these commits, the CPU backend would not have produced nontemporal instructions. The `cache_modifier` was stored on `triton::LoadOp`/`triton::StoreOp` but was never used in the CPU lowering pipeline.

---

## 5. Commit Timeline

| Commit | File | Change |
|--------|------|--------|
| **2bd6814b** | `MemoryOpToLLVM.cpp` | Added `nonTemporal = (op.getCache() == CacheModifier::CG)` and passed it to `LLVM::LoadOp`/`LLVM::StoreOp`. |
| **63d27de** | `ConvertMemoryOps.cpp` | Added `nt = (loadOp.getCache() == CacheModifier::CG)` and passed `/*nontemporal=*/nt` to `vector::LoadOp`/`vector::StoreOp`. |

---

## 6. Other Backends (Reference)

| Backend | Nontemporal handling |
|---------|------------------------|
| **NVIDIA (PTX)** | `LoadStoreOpToLLVM.cpp` uses `.cg` for cache modifier; emits PTX cache hints, not x86 nontemporal. |
| **AMD** | `Utility.cpp` maps `CacheModifier::CG` to nontemporal flags for predicated loads/stores. |

---

## 7. Appendix: File Reference

| File | Role |
|------|------|
| `third_party/cpu/lib/TritonToTritonCPU/ConvertMemoryOps.cpp` | Propagates `CacheModifier::CG` → `nontemporal=true` on `vector::LoadOp`/`vector::StoreOp` |
| `third_party/cpu/lib/TritonCPUToLLVM/MemoryOpToLLVM.cpp` | Propagates `CacheModifier::CG` → `nonTemporal=true` on `LLVM::LoadOp`/`LLVM::StoreOp` |
| `third_party/cpu/lib/TritonToTritonCPU/ConvertNUMAOps.cpp` | Assigns `CacheModifier::CG` or `CA` to NUMA loads/stores via `analyzeReuseDistance()` |
| `python/triton/language/semantic.py` | Maps `cache_modifier=".cg"` to `CacheModifier::CG` for user-facing `tl.load`/`tl.store` |
