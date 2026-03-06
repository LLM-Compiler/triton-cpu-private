#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

struct ConvertNUMAOps
    : public triton::impl::ConvertNUMAOpsBase<ConvertNUMAOps> {
  using ConvertNUMAOpsBase::ConvertNUMAOpsBase;

  ConvertNUMAOps() : ConvertNUMAOpsBase() {}

  void runOnOperation() override {
    // This pass is a placeholder for TritonDialect->TritonCPU conversion
    // of NUMA operations. Currently, the NUMA ops are defined directly in the
    // TritonCPU dialect, so this pass simply legalizes the GetNumaIdOp by
    // leaving it in place for the downstream TritonCPUToLLVM pass to handle.
    //
    // Future work: implement more complex lowering logic here, e.g.,
    // decomposing make_numa_buffer into alloc + numa_bind sequences.
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
