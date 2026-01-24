//===----------------------------------------------------------------------===//
// Epilogue Fusion Pass
//===----------------------------------------------------------------------===//
//
// This pass fuses dot product operations with subsequent bias addition
// operations. It transforms patterns like:
//
//   %dot_result = triton_cpu.dot %a, %b, %c
//   %result = arith.addf %dot_result, %bias
//
// Into:
//
//   %fused_acc = arith.addf %c, %bias
//   %result = triton_cpu.dot %a, %b, %fused_acc
//
// Optimization Benefits:
// 1. Reduces memory traffic by eliminating intermediate stores/loads
// 2. Enables better register utilization (keeps values in registers)
// 3. Improves cache locality (fewer memory accesses)
// 4. Enables hardware FMA optimizations in subsequent passes
// 5. Reduces instruction count (one operation instead of two)
//
// The pass runs in the make_tttcir stage, after mask optimization and before
// dot-to-hardware conversions, allowing the fused pattern to benefit from
// hardware-specific optimizations.
//
//===----------------------------------------------------------------------===//

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_EPILOGUEFUSION
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

//===----------------------------------------------------------------------===//
// FuseDotBiasPattern: Float Addition Fusion
//===----------------------------------------------------------------------===//
//
// This pattern matches and rewrites:
//   Original: arith.addf(triton_cpu.dot(a, b, c), bias)
//   Fused:    triton_cpu.dot(a, b, arith.addf(c, bias))
//
// The transformation is mathematically equivalent:
//   Before: result = (A × B) + bias
//   After:  result = (A × B) + (c + bias)
//
// If c is zero (common accumulator initialization), this simplifies to:
//   After:  result = (A × B) + bias (same result, better performance)
//
// Safety: Only fuses when the dot result has exactly one use to avoid
//         breaking other consumers of the dot operation.
//
//===----------------------------------------------------------------------===//
struct FuseDotBiasPattern : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  FuseDotBiasPattern(MLIRContext *context)
      : OpRewritePattern<arith::AddFOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                PatternRewriter &rewriter) const override {
    // Step 1: Extract operands from the addition operation
    // Addition is commutative, so we need to check both operands
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    // Step 2: Identify which operand is the dot product result
    // Try lhs first, then rhs (since addition is commutative)
    cpu::DotOp dotOp = lhs.getDefiningOp<cpu::DotOp>();
    Value bias = rhs;

    if (!dotOp) {
      // Try the other way: maybe bias is on the left, dot on the right
      dotOp = rhs.getDefiningOp<cpu::DotOp>();
      bias = lhs;
    }

    // Pattern doesn't match if neither operand is a dot product result
    if (!dotOp) {
      return failure();
    }

    // Step 3: Safety check - ensure the dot result has only one use
    // This is critical: if the dot result is used elsewhere, fusing would
    // break those other uses. We can only fuse when the dot result is
    // exclusively used by this addition operation.
    if (!dotOp.getResult().hasOneUse()) {
      return failure();
    }

    // Step 4: Extract the original accumulator from the dot operation
    // The accumulator 'c' in dot(a, b, c) is where we'll fuse the bias
    Value originalAcc = dotOp.getC();

    // Step 5: Create the fused accumulator by adding bias to the original accumulator
    // Insert this operation before the dot to ensure proper SSA ordering
    rewriter.setInsertionPoint(dotOp);
    
    // Create: fused_acc = original_acc + bias
    // This will be used as the new accumulator in the fused dot operation
    Value fusedAcc = rewriter.create<arith::AddFOp>(
        addOp.getLoc(), originalAcc, bias);

    // Step 6: Create the new fused dot operation
    // This replaces the old dot with one that uses the fused accumulator
    // Insert after the old dot to maintain operation order
    rewriter.setInsertionPointAfter(dotOp);
    
    auto fusedDot = rewriter.create<cpu::DotOp>(
        dotOp.getLoc(),                    // Preserve source location for debugging
        addOp.getType(),                   // Result type matches the addition result
        dotOp.getA(),                      // Preserve operand A
        dotOp.getB(),                      // Preserve operand B
        fusedAcc,                          // Use fused accumulator (c + bias)
        dotOp.getInputPrecision(),         // Preserve input precision attribute
        dotOp.getMaxNumImpreciseAcc()      // Preserve max imprecise accumulator setting
    );

    // Step 7: Replace the addition operation with the fused dot result
    // This tells MLIR: wherever 'addOp' was used, use 'fusedDot' result instead
    rewriter.replaceOp(addOp, fusedDot.getResult());

    // The original dotOp will be automatically removed by Dead Code Elimination (DCE)
    // because it now has zero users (we replaced its only use with the fused dot)

    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseDotBiasIntPattern: Integer Addition Fusion
//===----------------------------------------------------------------------===//
//
// Same as FuseDotBiasPattern but for integer addition operations.
// Matches: arith.addi(triton_cpu.dot(a, b, c), bias)
// Fuses:   triton_cpu.dot(a, b, arith.addi(c, bias))
//
// This pattern handles integer matrix operations, which are less common
// but still important for certain workloads (e.g., quantized neural networks).
//
//===----------------------------------------------------------------------===//
struct FuseDotBiasIntPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  FuseDotBiasIntPattern(MLIRContext *context)
      : OpRewritePattern<arith::AddIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    cpu::DotOp dotOp = lhs.getDefiningOp<cpu::DotOp>();
    Value bias = rhs;

    if (!dotOp) {
      dotOp = rhs.getDefiningOp<cpu::DotOp>();
      bias = lhs;
    }

    if (!dotOp) {
      return failure();
    }

    if (!dotOp.getResult().hasOneUse()) {
      return failure();
    }

    Value originalAcc = dotOp.getC();
    rewriter.setInsertionPoint(dotOp);
    
    Value fusedAcc = rewriter.create<arith::AddIOp>(
        addOp.getLoc(), originalAcc, bias);

    rewriter.setInsertionPointAfter(dotOp);
    
    auto fusedDot = rewriter.create<cpu::DotOp>(
        dotOp.getLoc(),
        addOp.getType(),
        dotOp.getA(),
        dotOp.getB(),
        fusedAcc,
        dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc()
    );

    rewriter.replaceOp(addOp, fusedDot.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// EpilogueFusionPass: Main Pass Implementation
//===----------------------------------------------------------------------===//
//
// This pass applies the epilogue fusion patterns to the entire module.
// It uses MLIR's greedy pattern rewriter which:
// 1. Applies patterns in order of benefit (higher benefit = higher priority)
// 2. Repeats until no more patterns match (convergence)
// 3. Handles pattern interactions automatically
//
// The pass runs on ModuleOp, meaning it processes all functions in the module.
//
//===----------------------------------------------------------------------===//
struct EpilogueFusionPass
    : public triton::cpu::impl::EpilogueFusionBase<EpilogueFusionPass> {
  EpilogueFusionPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Create a pattern set and add our fusion patterns
    RewritePatternSet patterns(context);
    patterns.add<FuseDotBiasPattern>(context);        // Float fusion
    patterns.add<FuseDotBiasIntPattern>(context);      // Integer fusion
    
    // Apply patterns greedily until convergence
    // This will repeatedly apply patterns until no more matches are found
    // The greedy rewriter handles:
    // - Pattern ordering (by benefit)
    // - Dead code elimination
    // - Pattern interaction
    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createEpilogueFusionPass() {
  return std::make_unique<EpilogueFusionPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir

