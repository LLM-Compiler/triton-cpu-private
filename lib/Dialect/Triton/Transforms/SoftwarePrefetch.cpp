#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h" // Required for Pass
#include "mlir/Dialect/Func/IR/FuncOps.h" // Required for func::FuncOp

// Required to use the TableGen'd pass base class
#define GEN_PASS_DEF_SOFTWAREPREFETCH
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

using namespace mlir;

namespace mlir {
namespace triton {
namespace {

struct SoftwarePrefetchPass : public impl::TritonSoftwarePrefetchBase<SoftwarePrefetchPass> {
  // Use the option defined in TableGen
  using impl::TritonSoftwarePrefetchBase<SoftwarePrefetchPass>::prefetchDistance;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    // Debug print to confirm pass is running
    // llvm::errs() << "[TritonCPU] Running Software Prefetch Pass...\n";

    // Walk every loop in the function
    func.walk([&](scf::ForOp forOp) {
      injectPrefetch(forOp);
    });
  }

  void injectPrefetch(scf::ForOp forOp) {
    OpBuilder builder(forOp.getBody()->getTerminator());
    Location loc = forOp.getLoc();

    Value iv = forOp.getInductionVar();
    Value distConst = builder.create<arith::ConstantIndexOp>(loc, prefetchDistance);

    // Scan for Load Operations
    forOp.getBody()->walk([&](memref::LoadOp loadOp) {
      
      bool dependsOnIV = false;
      SmallVector<Value> newIndices;
      
      // Check indices: If index == InductionVar, we prefetch (IV + Distance)
      for (Value index : loadOp.getIndices()) {
        if (index == iv) {
          Value newIdx = builder.create<arith::AddIOp>(loc, iv, distConst);
          newIndices.push_back(newIdx);
          dependsOnIV = true;
        } else {
          newIndices.push_back(index);
        }
      }

      if (dependsOnIV) {
        // Insert prefetch: Read(0), Locality(3=L1), DataCache(1)
        builder.create<memref::PrefetchOp>(
            loc, 
            loadOp.getMemRef(), 
            newIndices, 
            /*isWrite=*/false, 
            /*locality=*/3, 
            /*isDataCache=*/true
        );
      }
    });
  }
};

} // namespace
} // namespace triton
} // namespace mlir

std::unique_ptr<Pass> mlir::triton::createSoftwarePrefetchPass() {
  return std::make_unique<SoftwarePrefetchPass>();
}
