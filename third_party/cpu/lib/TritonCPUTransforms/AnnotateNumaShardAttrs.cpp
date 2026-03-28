#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include <cstdlib>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_ANNOTATENUMASHARDATTRS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct AnnotateNumaShardAttrs
    : public triton::cpu::impl::AnnotateNumaShardAttrsBase<
          AnnotateNumaShardAttrs> {
  using AnnotateNumaShardAttrsBase::AnnotateNumaShardAttrsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    int64_t shardCount = 1;
    if (const char *env = std::getenv("TRITON_CPU_NUMA_SHARDS")) {
      char *end = nullptr;
      long parsed = std::strtol(env, &end, 10);
      if (end != env && parsed > 0) {
        shardCount = static_cast<int64_t>(parsed);
      }
    }

    mod.walk([&](cpu::DotOp op) {
      if (!op->hasAttr("ttcpu.numa.shard_count")) {
        op->setAttr("ttcpu.numa.shard_count", IntegerAttr::get(
                                                  IntegerType::get(mod.getContext(), 64),
                                                  shardCount));
      }
      if (!op->hasAttr("ttcpu.numa.partition_axis")) {
        // Convention: 1 -> column split for first GEMM, 0 -> row split for second GEMM.
        // The final role is supplied by upstream FX metadata/launch metadata.
        op->setAttr("ttcpu.numa.partition_axis", IntegerAttr::get(
                                                      IntegerType::get(mod.getContext(), 64),
                                                      1));
      }
    });
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateNumaShardAttrs() {
  return std::make_unique<AnnotateNumaShardAttrs>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
