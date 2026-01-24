# Testing and Verifying the Epilogue Fusion Pass

This guide explains how to test and verify that the Epilogue Fusion pass is working correctly.

## Table of Contents

1. [Building the Pass](#building-the-pass)
2. [Verifying Pass Registration](#verifying-pass-registration)
3. [Testing with MLIR File](#testing-with-mlir-file)
4. [Writing Unit Tests](#writing-unit-tests)
5. [Integration Testing](#integration-testing)
6. [Debugging the Pass](#debugging-the-pass)
7. [Verification Checklist](#verification-checklist)

## Building the Pass

### Step 1: Build the Project

First, ensure the pass is compiled:

```bash
cd /home/shavin/llm-cpu-compiler/triton-cpu
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) TritonCPUTransforms
```

### Step 2: Verify Compilation

Check that `FuseDotBias.cpp` is compiled:

```bash
# Check if the object file exists
find . -name "FuseDotBias.o" -o -name "FuseDotBias.cpp.o"

# Or check the library
find . -name "*TritonCPUTransforms*" -type f
```

## Verifying Pass Registration

### Method 1: Check Pass Registration

The pass should be registered with MLIR. Verify this:

```bash
cd /home/shavin/llm-cpu-compiler/triton-cpu/build
# If you have mlir-opt or similar tools
./bin/mlir-opt --help | grep -i "epilogue"
```

### Method 2: Python API Check

Verify the pass is exposed in Python:

```python
import triton
import triton.language as tl
from triton.backends.cpu import cpu

# Check if the pass function exists
print(hasattr(cpu.passes.ttcpuir, 'add_epilogue_fusion'))
# Should print: True
```

## Testing with MLIR File

### Step 1: Create a Test MLIR File

Create a file `test_epilogue_fusion.mlir`:

```mlir
module {
  func.func @test_fusion(%arg0: vector<128x128xf32>, 
                          %arg1: vector<128x128xf32>,
                          %arg2: vector<128x128xf32>,
                          %bias: vector<128x128xf32>) -> vector<128x128xf32> {
    // This should be fused: dot + add
    %zero = arith.constant dense<0.0> : vector<128x128xf32>
    %dot_result = triton_cpu.dot %arg0, %arg1, %zero : 
        vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
    %result = arith.addf %dot_result, %bias : vector<128x128xf32>
    return %result : vector<128x128xf32>
  }
}
```

### Step 2: Run the Pass

If you have `mlir-opt` or similar:

```bash
# Run just the epilogue fusion pass
mlir-opt test_epilogue_fusion.mlir \
  --triton-cpu-epilogue-fusion \
  --canonicalize \
  --print-ir-after-all
```

### Step 3: Verify Transformation

The output should show the fused pattern:

```mlir
// Before (original):
%dot_result = triton_cpu.dot %arg0, %arg1, %zero
%result = arith.addf %dot_result, %bias

// After (fused):
%fused_acc = arith.addf %zero, %bias
%result = triton_cpu.dot %arg0, %arg1, %fused_acc
```

## Writing Unit Tests

### C++ Unit Test

Create `test/FuseDotBiasTest.cpp`:

```cpp
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

class EpilogueFusionTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect, 
                        func::FuncDialect,
                        TritonCPUDialect>();
  }

  MLIRContext context;
};

TEST_F(EpilogueFusionTest, BasicFusion) {
  const char *mlirCode = R"(
    module {
      func.func @test(%a: vector<128x128xf32>, 
                      %b: vector<128x128xf32>,
                      %bias: vector<128x128xf32>) -> vector<128x128xf32> {
        %zero = arith.constant dense<0.0> : vector<128x128xf32>
        %dot = triton_cpu.dot %a, %b, %zero : 
            vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
        %result = arith.addf %dot, %bias : vector<128x128xf32>
        return %result : vector<128x128xf32>
      }
    }
  )";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(mlirCode, &context);
  ASSERT_TRUE(module);

  // Run the pass
  PassManager pm(&context);
  pm.addPass(createEpilogueFusionPass());
  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Verify: Check that the add operation is gone and dot uses fused accumulator
  module->walk([&](arith::AddFOp addOp) {
    // Should not find any addf operations (they should be fused)
    FAIL() << "Found unfused addf operation";
  });

  // Verify: Check that dot operations exist with fused accumulators
  bool foundFusedDot = false;
  module->walk([&](cpu::DotOp dotOp) {
    Value acc = dotOp.getC();
    // Check if accumulator is an add operation (fused)
    if (acc.getDefiningOp<arith::AddFOp>()) {
      foundFusedDot = true;
    }
  });
  
  ASSERT_TRUE(foundFusedDot) << "Fusion did not occur";
}

TEST_F(EpilogueFusionTest, NoFusionWhenMultipleUses) {
  const char *mlirCode = R"(
    module {
      func.func @test(%a: vector<128x128xf32>, 
                      %b: vector<128x128xf32>,
                      %bias: vector<128x128xf32>) -> (vector<128x128xf32>, vector<128x128xf32>) {
        %zero = arith.constant dense<0.0> : vector<128x128xf32>
        %dot = triton_cpu.dot %a, %b, %zero : 
            vector<128x128xf32> * vector<128x128xf32> -> vector<128x128xf32>
        %result1 = arith.addf %dot, %bias : vector<128x128xf32>
        %result2 = arith.mulf %dot, %bias : vector<128x128xf32>  // Multiple uses!
        return %result1, %result2 : vector<128x128xf32>, vector<128x128xf32>
      }
    }
  )";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(mlirCode, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(createEpilogueFusionPass());
  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Should NOT fuse because dot has multiple uses
  int addCount = 0;
  module->walk([&](arith::AddFOp addOp) {
    addCount++;
  });
  
  // Should still have the add operation (not fused)
  ASSERT_GT(addCount, 0) << "Add operation should remain when dot has multiple uses";
}
```

### Running C++ Tests

Add to `test/CMakeLists.txt`:

```cmake
add_mlir_test(TritonCPUTransformsFuseDotBiasTest
  FuseDotBiasTest.cpp
)
target_link_libraries(TritonCPUTransformsFuseDotBiasTest
  PRIVATE
  TritonCPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
  MLIRPass
  MLIRTestSupport
)
```

Run tests:

```bash
cd build
make TritonCPUTransformsFuseDotBiasTest
./bin/TritonCPUTransformsFuseDotBiasTest
```

## Integration Testing

### Python Integration Test

Create `test/test_epilogue_fusion.py`:

```python
import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def fused_matmul_bias(x, y, bias, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """
    Matrix multiplication with bias: out = x @ y + bias
    This should trigger the epilogue fusion pass.
    """
    pid = tl.program_id(0)
    # Load inputs
    x_ptr = x + pid * K
    y_ptr = y
    bias_ptr = bias + pid * N
    
    # Compute dot product
    acc = tl.zeros([N], dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(x_ptr + k)
        b = tl.load(y_ptr + k * N)
        acc += a * b
    
    # Add bias (this should be fused by the pass)
    acc = acc + tl.load(bias_ptr)
    
    # Store result
    out_ptr = out + pid * N
    tl.store(out_ptr, acc)

def test_epilogue_fusion():
    """Test that epilogue fusion works correctly"""
    M, N, K = 128, 64, 32
    
    # Create test data
    x = torch.randn(M, K, device='cpu', dtype=torch.float32)
    y = torch.randn(K, N, device='cpu', dtype=torch.float32)
    bias = torch.randn(N, device='cpu', dtype=torch.float32)
    out = torch.zeros(M, N, device='cpu', dtype=torch.float32)
    
    # Run kernel
    grid = (M,)
    fused_matmul_bias[grid](x, y, bias, out, M, N, K)
    
    # Verify correctness (compare with PyTorch)
    expected = torch.matmul(x, y) + bias.unsqueeze(0)
    
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Output mismatch! Max diff: {torch.max(torch.abs(out - expected))}"
    
    print("✓ Epilogue fusion test passed!")

if __name__ == "__main__":
    test_epilogue_fusion()
```

Run the test:

```bash
cd /home/shavin/llm-cpu-compiler/triton-cpu
python test/test_epilogue_fusion.py
```

## Debugging the Pass

### Method 1: Enable Debug Output

Add debug prints to the pass:

```cpp
// In FuseDotBias.cpp, add at the top:
#define DEBUG_TYPE "epilogue-fusion"
#include "mlir/Support/Debug.h"

// In matchAndRewrite:
LLVM_DEBUG(llvm::dbgs() << "EpilogueFusion: Matching add operation\n");
LLVM_DEBUG(llvm::dbgs() << "EpilogueFusion: Found dot operation\n");
LLVM_DEBUG(llvm::dbgs() << "EpilogueFusion: Fusing dot with bias\n");
```

Compile with debug:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON
make -j$(nproc) TritonCPUTransforms
```

Run with debug output:

```bash
MLIR_LLVM_DEBUG=epilogue-fusion mlir-opt test.mlir --triton-cpu-epilogue-fusion
```

### Method 2: Print IR Before/After

Use MLIR's print options:

```bash
mlir-opt test.mlir \
  --print-ir-before=triton-cpu-epilogue-fusion \
  --print-ir-after=triton-cpu-epilogue-fusion \
  --triton-cpu-epilogue-fusion
```

### Method 3: Use Python Debug Mode

In Python:

```python
import triton
from triton.backends.cpu import cpu

# Enable debug output
import os
os.environ['TRITON_CPU_DEBUG'] = '1'

# Your test code here
```

### Method 4: Verify in Compiler Pipeline

Check the compiler pipeline in `compiler.py`:

```python
# In compiler.py, add debug prints:
def make_tttcir(self, mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()  # Enable debug output
    cpu.passes.ttcpuir.add_triton_cpu_canonicalizer(pm)
    cpu.passes.ttcpuir.add_optimize_masks(pm)
    cpu.passes.ttcpuir.add_epilogue_fusion(pm)  # Your pass
    print("Running epilogue fusion pass...")  # Debug print
    pm.run(mod)
    return mod
```

## Verification Checklist

Use this checklist to verify the pass works:

### ✅ Build Verification
- [ ] Project compiles without errors
- [ ] `FuseDotBias.cpp` is included in build
- [ ] No linker errors

### ✅ Registration Verification
- [ ] Pass appears in `--help` output (if available)
- [ ] Python API exposes `add_epilogue_fusion`
- [ ] Pass can be instantiated without errors

### ✅ Pattern Matching Verification
- [ ] Pass matches `addf(dot, bias)` pattern
- [ ] Pass matches `addf(bias, dot)` pattern (commutative)
- [ ] Pass matches `addi(dot, bias)` pattern (integer)
- [ ] Pass correctly rejects patterns with multiple uses

### ✅ Transformation Verification
- [ ] Add operation is removed after fusion
- [ ] New dot operation is created with fused accumulator
- [ ] Original dot operation is removed (by DCE)
- [ ] Result type is preserved
- [ ] Attributes are preserved (inputPrecision, etc.)

### ✅ Correctness Verification
- [ ] Fused code produces same results as original
- [ ] No correctness regressions in existing tests
- [ ] Edge cases handled (zero accumulator, etc.)

### ✅ Performance Verification
- [ ] Memory traffic is reduced (verify with profiler)
- [ ] Instruction count is reduced
- [ ] No performance regressions

## Quick Verification Script

Create `verify_pass.sh`:

```bash
#!/bin/bash

echo "=== Building Pass ==="
cd /home/shavin/llm-cpu-compiler/triton-cpu/build
make -j$(nproc) TritonCPUTransforms || exit 1

echo "=== Checking Python API ==="
python3 -c "
from triton.backends.cpu import cpu
if hasattr(cpu.passes.ttcpuir, 'add_epilogue_fusion'):
    print('✓ Pass is exposed in Python API')
else:
    print('✗ Pass NOT found in Python API')
    exit(1)
"

echo "=== Running Simple Test ==="
python3 << 'PYTHON'
import triton
import triton.language as tl
import torch

@triton.jit
def test_kernel(x, y, bias, out, N: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([N], dtype=tl.float32)
    for i in range(N):
        acc[i] = x[i] * y[i]
    acc = acc + bias  # This should be fused
    tl.store(out + pid * N, acc)

# Test
N = 32
x = torch.randn(N)
y = torch.randn(N)
bias = torch.randn(N)
out = torch.zeros(N)

test_kernel[(1,)](x, y, bias, out, N)
print("✓ Kernel executed successfully")
PYTHON

echo "=== All checks passed! ==="
```

Make it executable and run:

```bash
chmod +x verify_pass.sh
./verify_pass.sh
```

## Common Issues and Solutions

### Issue: Pass not found
**Solution**: Check CMakeLists.txt includes the source file

### Issue: Pattern not matching
**Solution**: 
- Verify dot operation is `triton_cpu.dot` (not `triton.dot`)
- Check that dot result has exactly one use
- Enable debug output to see why pattern fails

### Issue: Transformation not happening
**Solution**:
- Check pass is actually being called in the pipeline
- Verify pass runs after dot conversion but before other optimizations
- Check for conflicting patterns

### Issue: Incorrect results
**Solution**:
- Verify mathematical equivalence
- Check type compatibility
- Ensure accumulator handling is correct

## Next Steps

After verification:
1. Add comprehensive unit tests
2. Add integration tests to test suite
3. Benchmark performance improvements
4. Document any limitations or edge cases found

