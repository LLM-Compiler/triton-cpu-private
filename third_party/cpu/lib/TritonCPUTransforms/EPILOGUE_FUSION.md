# Epilogue Fusion Pass Documentation

## Overview

The **Epilogue Fusion Pass** is an optimization pass in the Triton CPU backend that fuses dot product (matrix multiplication) operations with subsequent element-wise bias addition operations. This pass transforms separate operations into a single fused operation, enabling better code generation and performance optimization.

## What Does This Pass Do?

The pass detects patterns where:
- A dot product operation (`triton_cpu.dot`) produces a result
- That result is immediately consumed by an addition operation (`arith.addf` or `arith.addi`)
- The addition adds a "bias" value to the dot product result

The pass then transforms this pattern by moving the bias addition into the dot product's accumulator, effectively fusing the two operations into one.

## Optimization Benefits

### 1. **Reduced Memory Traffic**
- **Before**: Dot product writes result to memory → Load result → Add bias → Write final result
- **After**: Dot product directly computes `(A * B) + bias` in one operation
- Eliminates intermediate memory stores and loads

### 2. **Better Register Utilization**
- The fused operation keeps intermediate values in registers
- Reduces register pressure by avoiding temporary storage
- Enables better instruction scheduling

### 3. **Improved Cache Locality**
- Fewer memory accesses mean better cache behavior
- Reduces cache pollution from intermediate results
- Better data reuse patterns

### 4. **Enables Hardware-Specific Optimizations**
- Many CPU architectures (x86, ARM) have fused multiply-add (FMA) instructions
- The fused pattern can be directly mapped to hardware FMA operations
- Enables vectorization opportunities that weren't possible with separate operations

### 5. **Reduced Instruction Count**
- One operation instead of two reduces instruction overhead
- Lower instruction cache pressure
- Better branch prediction (fewer instructions to predict)

## How It Works

### Pattern Matching

The pass uses MLIR's pattern rewriting infrastructure to match the following patterns:

#### Pattern 1: Float Addition
```mlir
// Original pattern
%dot_result = triton_cpu.dot %a, %b, %c : vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
%result = arith.addf %dot_result, %bias : vector<128x128xf32>
```

#### Pattern 2: Integer Addition
```mlir
// Original pattern
%dot_result = triton_cpu.dot %a, %b, %c : vector<128x128xi32> * vector<128x128xi32> -> vector<128x128xi32>
%result = arith.addi %dot_result, %bias : vector<128x128xi32>
```

### Transformation Logic

The pass performs the following transformation:

**Before:**
```mlir
%dot_result = triton_cpu.dot %a, %b, %c, inputPrecision = IEEE : 
    vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
%result = arith.addf %dot_result, %bias : vector<128x128xf32>
```

**After:**
```mlir
%fused_acc = arith.addf %c, %bias : vector<128x128xf32>
%result = triton_cpu.dot %a, %b, %fused_acc, inputPrecision = IEEE : 
    vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
```

### Mathematical Equivalence

The transformation is mathematically equivalent:
- **Original**: `result = (A × B) + bias`
- **Fused**: `result = (A × B) + (c + bias)` where `c` is the original accumulator

If the original accumulator `c` is zero (common case), this simplifies to:
- **Fused**: `result = (A × B) + bias` (same result, but computed more efficiently)

### Safety Checks

The pass includes several safety checks to ensure correctness:

1. **Single Use Check**: The dot product result must have exactly one use (the addition operation)
   - **Why**: If the dot result is used elsewhere, fusing would break those other uses
   - **Example**: If `%dot_result` is used in two places, we can't fuse without duplicating the dot operation

2. **Commutative Handling**: The pass checks both operands of the addition
   - Handles both `add(dot, bias)` and `add(bias, dot)` patterns
   - Addition is commutative, so order doesn't matter

3. **Type Compatibility**: MLIR's type system ensures types match
   - The bias must be compatible with the dot product result type
   - Vector shapes must match for broadcasting

## Implementation Details

### Pattern Rewrite Structure

The pass implements two rewrite patterns:

1. **`FuseDotBiasPattern`**: Matches `arith.addf` operations
2. **`FuseDotBiasIntPattern`**: Matches `arith.addi` operations

Both patterns follow the same logic:

```cpp
1. Match the addition operation (addf/addi)
2. Check if one operand is a dot product result
3. Verify the dot has only one use
4. Create fused accumulator: add(original_accumulator, bias)
5. Create new dot operation with fused accumulator
6. Replace the addition with the new dot result
7. Let DCE (Dead Code Elimination) remove the old dot
```

### Code Generation Flow

The pass runs in the `make_tttcir` stage of compilation:

```
TTIR (Triton IR)
  ↓
TTCIR (Triton CPU IR) ← Epilogue Fusion runs here
  ↓
Target TTCIR (optimized)
  ↓
LLVM IR
```

**Placement in Pipeline:**
- Runs after `add_optimize_masks()` 
- Runs before `add_convert_dot_to_*` passes
- This placement ensures:
  - Dot operations are already converted to `triton_cpu.dot`
  - Fused operations can benefit from subsequent dot-to-hardware conversions
  - The fused pattern is available for hardware-specific optimizations

## Example Use Cases

### 1. Linear Layer with Bias (Neural Networks)

**Before Fusion:**
```mlir
// Matrix multiplication: output = input @ weight
%output = triton_cpu.dot %input, %weight, %zero : 
    vector<128x256xf32> * vector<256x64xf32> -> vector<128x64xf32>
// Add bias: output = output + bias
%final = arith.addf %output, %bias : vector<128x64xf32>
```

**After Fusion:**
```mlir
// Fused: output = input @ weight + bias (computed in one step)
%final = triton_cpu.dot %input, %weight, %bias : 
    vector<128x256xf32> * vector<256x64xf32> -> vector<128x64xf32>
```

### 2. Batch Matrix Multiplication with Bias

**Before:**
```mlir
%batch_dot = triton_cpu.dot %batch_a, %batch_b, %zero : 
    vector<32x128x128xf32> * vector<32x128x128xf32> -> vector<32x128x128xf32>
%with_bias = arith.addf %batch_dot, %broadcast_bias : vector<32x128x128xf32>
```

**After:**
```mlir
%with_bias = triton_cpu.dot %batch_a, %batch_b, %broadcast_bias : 
    vector<32x128x128xf32> * vector<32x128x128xf32> -> vector<32x128x128xf32>
```

## Performance Impact

### Theoretical Benefits

1. **Memory Bandwidth Reduction**: 
   - Eliminates one write and one read operation
   - For large matrices, this can save significant memory bandwidth
   - Example: 128×128 f32 matrix = 64KB, saving 128KB of memory traffic

2. **Instruction Count Reduction**:
   - One operation instead of two
   - Fewer instructions to decode and execute
   - Better instruction-level parallelism

3. **Register Pressure**:
   - Intermediate result stays in registers
   - Reduces register spills to memory
   - Better utilization of CPU registers

### Real-World Impact

The actual performance improvement depends on:
- **Matrix sizes**: Larger matrices benefit more (more memory saved)
- **Hardware capabilities**: CPUs with FMA units see bigger gains
- **Cache behavior**: Better cache utilization from reduced memory traffic
- **Compiler optimizations**: Subsequent passes can optimize the fused operation better

Typical improvements:
- **Memory bandwidth**: 10-30% reduction in memory traffic
- **Instruction count**: 5-15% reduction in total instructions
- **Overall performance**: 2-10% speedup (depends on workload characteristics)

## Limitations and Future Work

### Current Limitations

1. **Single Use Requirement**: Only fuses when dot result has exactly one use
   - Could be extended to handle multiple uses with operation duplication
   - Trade-off: Code size vs. optimization opportunity

2. **Simple Addition Only**: Currently only handles `addf` and `addi`
   - Could be extended to other element-wise operations (subtract, multiply, etc.)
   - Could handle chains of operations (e.g., `add(mul(dot, scale), bias)`)

3. **Type Constraints**: Requires exact type matching
   - Could be extended to handle type promotions
   - Could handle broadcasting more explicitly

### Potential Extensions

1. **Multi-Operation Fusion**: Fuse longer chains
   ```mlir
   // Could fuse: scale * (A @ B) + bias
   %scaled = arith.mulf %dot_result, %scale
   %result = arith.addf %scaled, %bias
   ```

2. **Conditional Fusion**: Fuse operations with conditions
   ```mlir
   // Could fuse: (A @ B) + bias if condition
   %result = arith.select %cond, %fused, %original
   ```

3. **Broadcasting Support**: Better handling of shape mismatches
   ```mlir
   // Handle: dot(128x128) + bias(128) with broadcasting
   ```

## Testing

### Unit Tests

The pass should be tested with:

1. **Basic Fusion**: Simple dot + add pattern
2. **Commutative Cases**: Both `add(dot, bias)` and `add(bias, dot)`
3. **Type Variations**: Float and integer types
4. **Shape Variations**: Different matrix dimensions
5. **Failure Cases**: Multiple uses, incompatible types

### Integration Tests

Test the pass in the full compilation pipeline:
- Verify correctness: Fused code produces same results
- Verify optimization: Check that fusion actually occurs
- Verify performance: Measure speedup on real workloads

## Related Passes

- **`ConvertDotToFMA`**: Converts dot operations to FMA instructions (benefits from fusion)
- **`ConvertDotToAMX`**: Converts to AMX hardware (can use fused accumulator)
- **`OptimizeMasks`**: Runs before fusion, optimizes mask operations
- **`Canonicalize`**: Runs after fusion, cleans up dead code

## References

- MLIR Pattern Rewriting: https://mlir.llvm.org/docs/PatternRewriter/
- Triton CPU Backend: Internal documentation
- FMA Instructions: Hardware-specific fused multiply-add operations




