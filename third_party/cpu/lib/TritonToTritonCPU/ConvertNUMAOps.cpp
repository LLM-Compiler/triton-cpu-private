#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Default L3 cache capacity heuristic per socket (30 MB).
static constexpr int64_t kDefaultL3CapacityBytes = 30 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Polyhedral Reuse Analysis Helpers
// ---------------------------------------------------------------------------

static std::optional<int64_t> getConstantIntValue(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

static int64_t estimateTileBytes(Type resultTy) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy)) {
    int64_t numElems = 1;
    for (auto dim : tensorTy.getShape())
      numElems *= dim;
    auto elemTy = tensorTy.getElementType();
    if (elemTy.isIntOrFloat())
      return numElems * (elemTy.getIntOrFloatBitWidth() / 8);
    return numElems * 4;
  }
  if (resultTy.isIntOrFloat())
    return resultTy.getIntOrFloatBitWidth() / 8;
  return 4;
}

// Analyze the memory access pattern of a NumaLoadOp within its enclosing
// loop nests to determine the appropriate CacheModifier.
//
// Decision hierarchy:
//   1. Explicit frontend hint (eviction_policy attribute)  -> direct mapping
//   2. Loop-invariant pointer (same tile every iteration)  -> CA
//   3. Footprint within innermost loop fits in L3          -> CA
//   4. Default (streaming / unknown)                       -> CG
static CacheModifier analyzeReuseDistance(cpu::NumaLoadOp op) {
  // --- Step 1: Explicit frontend eviction-policy hint ---
  if (auto policyAttr = op->getAttrOfType<StringAttr>("eviction_policy")) {
    auto policy = policyAttr.getValue();
    if (policy == "evict_first")
      return CacheModifier::CG;
    if (policy == "evict_last")
      return CacheModifier::CA;
  }

  // --- Step 2: Collect enclosing scf::ForOp loops (innermost first) ---
  SmallVector<scf::ForOp> enclosingLoops;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      enclosingLoops.push_back(forOp);
    parent = parent->getParentOp();
  }

  // No enclosing loops -> one-shot remote fetch, stream without caching.
  if (enclosingLoops.empty())
    return CacheModifier::CG;

  // --- Step 3: Loop-invariance check ---
  // If the pointer is defined outside the outermost enclosing loop, the
  // same tile is re-loaded every iteration -> high temporal reuse.
  Value ptr = op.getPtr();
  auto outermostLoop = enclosingLoops.back();
  if (outermostLoop.isDefinedOutsideOfLoop(ptr))
    return CacheModifier::CA;

  // --- Step 4: Footprint estimation in the innermost loop ---
  // Compute  footprint = tripCount * tileBytes  and compare against L3.
  auto innermostLoop = enclosingLoops.front();
  auto lbVal = getConstantIntValue(innermostLoop.getLowerBound());
  auto ubVal = getConstantIntValue(innermostLoop.getUpperBound());
  auto stepVal = getConstantIntValue(innermostLoop.getStep());

  if (lbVal && ubVal && stepVal && *stepVal > 0) {
    int64_t tripCount = (*ubVal - *lbVal + *stepVal - 1) / *stepVal;
    int64_t tileBytes = estimateTileBytes(op.getResult().getType());
    int64_t footprint = tripCount * tileBytes;
    if (footprint <= kDefaultL3CapacityBytes)
      return CacheModifier::CA;
  }

  // Default: streaming bypass for remote NUMA data.
  return CacheModifier::CG;
}

// ---------------------------------------------------------------------------
// Conversion Target
// ---------------------------------------------------------------------------

class NUMAOpsConversionTarget : public ConversionTarget {
public:
  explicit NUMAOpsConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<arith::ArithDialect>();

    addLegalOp<triton::GetNumaIdOp>();

    // NumaLoadOp / NumaStoreOp must be rewritten to triton::LoadOp / StoreOp
    // so the standard scalarize -> memory-op-conversion pipeline handles them.
    addIllegalOp<cpu::NumaLoadOp>();
    addIllegalOp<cpu::NumaStoreOp>();
  }
};

// ---------------------------------------------------------------------------
// NumaLoadOp -> triton::LoadOp  (with polyhedral-derived cache modifier)
// ---------------------------------------------------------------------------

struct NumaLoadToLoadConversion : public OpRewritePattern<cpu::NumaLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cpu::NumaLoadOp op,
                                PatternRewriter &rewriter) const override {
    CacheModifier cache = analyzeReuseDistance(op);
    EvictionPolicy evict = (cache == CacheModifier::CG)
                               ? EvictionPolicy::EVICT_FIRST
                               : EvictionPolicy::NORMAL;

    Value mask = op.getMask();
    Value other = op.getOther();

    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, op.getPtr(), mask, other, cache, evict, /*isVolatile=*/false);
    return success();
  }
};

// ---------------------------------------------------------------------------
// NumaStoreOp -> triton::StoreOp  (always CG for remote stores)
// ---------------------------------------------------------------------------

struct NumaStoreToStoreConversion : public OpRewritePattern<cpu::NumaStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cpu::NumaStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value mask = op.getMask();

    rewriter.replaceOpWithNewOp<triton::StoreOp>(op, op.getPtr(),
                                                 op.getValue(), mask,
                                                 CacheModifier::CG,
                                                 EvictionPolicy::EVICT_FIRST);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pass Definition
// ---------------------------------------------------------------------------

struct ConvertNUMAOps
    : public triton::impl::ConvertNUMAOpsBase<ConvertNUMAOps> {
  using ConvertNUMAOpsBase::ConvertNUMAOpsBase;

  ConvertNUMAOps() : ConvertNUMAOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    NUMAOpsConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);
    patterns.add<NumaLoadToLoadConversion>(context);
    patterns.add<NumaStoreToStoreConversion>(context);

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
