#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTNUMAOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class NUMAOpsConversionTarget : public ConversionTarget {
public:
  explicit NUMAOpsConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    // Allow all Triton and TritonCPU dialect ops to remain.
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<arith::ArithDialect>();

    // Mark the Triton-dialect GetNumaIdOp as explicitly legal so it
    // passes through to the NUMAOpsToLLVM pass unchanged.
    addLegalOp<triton::GetNumaIdOp>();

    // All TritonCPU NUMA ops (MakeNumaBufferOp, NumaLoadOp, NumaStoreOp,
    // NumaWaitOp, NumaNotifyOp) are already legal because we added
    // TritonCPUDialect as a legal dialect above.
  }
};

struct ConvertNUMAOps
    : public triton::impl::ConvertNUMAOpsBase<ConvertNUMAOps> {
  using ConvertNUMAOpsBase::ConvertNUMAOpsBase;

  ConvertNUMAOps() : ConvertNUMAOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    NUMAOpsConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);

    // No conversion patterns needed: all NUMA ops are already in their
    // target dialect. This pass exists to let the pass pipeline manager
    // know that NUMA ops are legal at this stage, and to serve as the
    // future hook point for decomposition patterns (e.g., breaking
    // make_numa_buffer into alloc + bind sequences).

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertNUMAOps() {
  return std::make_unique<ConvertNUMAOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
