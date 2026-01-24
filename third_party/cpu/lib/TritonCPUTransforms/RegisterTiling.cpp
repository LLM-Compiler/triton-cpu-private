#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DEF_TRITONCPUREGISTERTILING
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

namespace {

struct RegisterTilingPattern : public OpRewritePattern<triton::cpu::DotOp> {
  RegisterTilingPattern(MLIRContext *context)
      : OpRewritePattern<triton::cpu::DotOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(triton::cpu::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto loc = dotOp.getLoc();
    auto aType = mlir::cast<VectorType>(dotOp.getA().getType());
    auto bType = mlir::cast<VectorType>(dotOp.getB().getType());
    auto cType = mlir::cast<VectorType>(dotOp.getC().getType());

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1]; // Assuming A is MxK
    int64_t N = cType.getShape()[1];

    // 1. Tiling Threshold & Configuration (Haswell / AVX2 Optimized)
    // We only tile if the block is larger than our target register pressure.
    // Haswell YMM = 8 floats.
    // Recommended Micro-tile: 8x4 (Use 4 Accumulator registers, leaving plenty for A/B loads)
    const int64_t tileM = 8; 
    const int64_t tileN = 4; 

    // Skip if the shape is already small enough to fit in registers
    if (M <= tileM && N <= tileN) {
      return failure();
    }

    // Ensure dimensions are divisible by tile size (Simplified for this snippet)
    // In production, you would handle remainders (peeling) or assume padding.
    if (M % tileM != 0 || N % tileN != 0) {
      // For now, fail on non-divisible shapes to avoid crash. 
      // Ideally, add padding or loop tail handling here.
      return failure(); 
    }

    Value initialC = dotOp.getC();
    Value inputA = dotOp.getA();
    Value inputB = dotOp.getB();

    // 2. Create Outer Loop (Iterate over M)
    auto stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    auto lbM = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ubM = rewriter.create<arith::ConstantIndexOp>(loc, M);

    // The outer loop carries the full Accumulator 'C'
    auto mLoop = rewriter.create<scf::ForOp>(
        loc, lbM, ubM, stepM, ValueRange{initialC},
        [&](OpBuilder &b, Location loc, Value ivM, ValueRange args) {
          Value carriedC_M = args[0];

          // 3. Create Inner Loop (Iterate over N)
          auto stepN = b.create<arith::ConstantIndexOp>(loc, tileN);
          auto lbN = b.create<arith::ConstantIndexOp>(loc, 0);
          auto ubN = b.create<arith::ConstantIndexOp>(loc, N);

          // The inner loop also carries 'C' to update it tile by tile
          auto nLoop = b.create<scf::ForOp>(
              loc, lbN, ubN, stepN, ValueRange{carriedC_M},
              [&](OpBuilder &bInner, Location locInner, Value ivN, ValueRange innerArgs) {
                Value carriedC_N = innerArgs[0];

                // --- MICRO KERNEL LOGIC STARTS HERE ---

                // 4. Extract Tile A (Size: tileM x K)
                // Offsets: [ivM, 0], Sizes: [tileM, K], Strides: [1, 1]
                auto subA = bInner.create<vector::ExtractStridedSliceOp>(
                    locInner, inputA, 
                    /*offsets=*/ValueRange{ivM, bInner.create<arith::ConstantIndexOp>(locInner, 0)}, 
                    /*sizes=*/ArrayRef<int64_t>{tileM, K}, 
                    /*strides=*/ArrayRef<int64_t>{1, 1});

                // 5. Extract Tile B (Size: K x tileN)
                // Offsets: [0, ivN], Sizes: [K, tileN], Strides: [1, 1]
                auto subB = bInner.create<vector::ExtractStridedSliceOp>(
                    locInner, inputB, 
                    /*offsets=*/ValueRange{bInner.create<arith::ConstantIndexOp>(locInner, 0), ivN}, 
                    /*sizes=*/ArrayRef<int64_t>{K, tileN}, 
                    /*strides=*/ArrayRef<int64_t>{1, 1});

                // 6. Extract Tile C (Accumulator) (Size: tileM x tileN)
                // Offsets: [ivM, ivN], Sizes: [tileM, tileN], Strides: [1, 1]
                auto subC = bInner.create<vector::ExtractStridedSliceOp>(
                    locInner, carriedC_N, 
                    /*offsets=*/ValueRange{ivM, ivN}, 
                    /*sizes=*/ArrayRef<int64_t>{tileM, tileN}, 
                    /*strides=*/ArrayRef<int64_t>{1, 1});

                // 7. Perform the Micro-Dot Product
                // This creates a smaller DotOp that fits in registers!
                auto subRes = bInner.create<triton::cpu::DotOp>(
                    locInner, subC.getType(), subA, subB, subC, dotOp.getAllowedInPrecision());

                // 8. Insert the Result back into the Accumulator
                auto updatedC = bInner.create<vector::InsertStridedSliceOp>(
                    locInner, subRes, carriedC_N, 
                    /*offsets=*/ValueRange{ivM, ivN}, 
                    /*strides=*/ArrayRef<int64_t>{1, 1});

                // Yield the updated C for the next iteration of N loop
                bInner.create<scf::YieldOp>(locInner, ValueRange{updatedC});
              });

          // Yield the updated C (after all N tiles done) for the next iteration of M loop
          b.create<scf::YieldOp>(loc, nLoop.getResults());
        });

    // 9. Replace the original massive DotOp with the result of the loops
    rewriter.replaceOp(dotOp, mLoop.getResult(0));

    return success();
  }
};

struct RegisterTilingPass
    : public triton::cpu::impl::TritonCPURegisterTilingBase<RegisterTilingPass> {
  RegisterTilingPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<RegisterTilingPattern>(context);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRegisterTilingPass() {
  return std::make_unique<RegisterTilingPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir