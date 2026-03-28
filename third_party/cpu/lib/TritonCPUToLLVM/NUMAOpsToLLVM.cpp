#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NUMAOPSTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

// Helper to get or insert a function declaration in the module.
static LLVM::LLVMFuncOp getOrInsertFuncDecl(OpBuilder &builder, ModuleOp mod,
                                              StringRef name, Type retTy,
                                              ArrayRef<Type> argTys) {
  if (auto existing = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
  return builder.create<LLVM::LLVMFuncOp>(mod.getLoc(), name, fnTy);
}

// Lower tt.get_numa_id → call @triton_get_numa_id() → i32
struct GetNumaIdOpConversion
    : public OpConversionPattern<triton::GetNumaIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumaIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcOp =
        getOrInsertFuncDecl(rewriter, mod, "triton_get_numa_id", i32Ty, {});
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), funcOp,
                                               ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// Lower ttcpu.make_numa_buffer → call @triton_numa_alloc(size, node) → ptr
struct MakeNumaBufferOpConversion
    : public OpConversionPattern<cpu::MakeNumaBufferOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::MakeNumaBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Compute total size in bytes from shape and element type bitwidth.
    auto shape = op.getStaticShape();
    int64_t numElements = 1;
    for (auto dim : shape)
      numElements *= dim;

    auto elementType = op.getElementType();
    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    int64_t totalBytes = numElements * (bitWidth / 8);

    auto sizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(totalBytes));
    auto nodeVal = adaptor.getNumaNode();

    auto funcOp = getOrInsertFuncDecl(rewriter, mod, "triton_numa_alloc",
                                       ptrTy, {i64Ty, i32Ty});
    auto call =
        rewriter.create<LLVM::CallOp>(loc, funcOp, ValueRange{sizeVal, nodeVal});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// Lower ttcpu.numa_wait → call @triton_numa_wait(signal_id)
struct NumaWaitOpConversion
    : public OpConversionPattern<cpu::NumaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::NumaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());

    auto signalVal = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getSignalId()));

    auto funcOp = getOrInsertFuncDecl(rewriter, mod, "triton_numa_wait",
                                       voidTy, {i32Ty});
    rewriter.create<LLVM::CallOp>(loc, funcOp, ValueRange{signalVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower ttcpu.numa_notify → call @triton_numa_notify(signal_id, node)
struct NumaNotifyOpConversion
    : public OpConversionPattern<cpu::NumaNotifyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::NumaNotifyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());

    auto signalVal = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getSignalId()));
    auto nodeVal = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getNumaDst()));

    auto funcOp = getOrInsertFuncDecl(rewriter, mod, "triton_numa_notify",
                                       voidTy, {i32Ty, i32Ty});
    rewriter.create<LLVM::CallOp>(loc, funcOp,
                                   ValueRange{signalVal, nodeVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// Safety-net fallback: any cpu::NumaLoadOp that survives to this pass
// indicates a pipeline ordering bug (ConvertNUMAOps should have rewritten
// it to triton::LoadOp before scalarize/memory-op-conversion).
struct NumaLoadOpFallback : public OpConversionPattern<cpu::NumaLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::NumaLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitOpError(
        "survived to NUMAOpsToLLVM -- it should have been converted to "
        "triton::LoadOp by the ConvertNUMAOps pass; check pipeline ordering");
  }
};

// Safety-net fallback for cpu::NumaStoreOp (same rationale as above).
struct NumaStoreOpFallback : public OpConversionPattern<cpu::NumaStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::NumaStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitOpError(
        "survived to NUMAOpsToLLVM -- it should have been converted to "
        "triton::StoreOp by the ConvertNUMAOps pass; check pipeline ordering");
  }
};

struct NUMAOpsToLLVM
    : public triton::impl::NUMAOpsToLLVMBase<NUMAOpsToLLVM> {
  using NUMAOpsToLLVMBase::NUMAOpsToLLVMBase;

  NUMAOpsToLLVM() : NUMAOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<GetNumaIdOpConversion>(typeConverter, context);
    patterns.add<MakeNumaBufferOpConversion>(typeConverter, context);
    patterns.add<NumaWaitOpConversion>(typeConverter, context);
    patterns.add<NumaNotifyOpConversion>(typeConverter, context);
    patterns.add<NumaLoadOpFallback>(typeConverter, context);
    patterns.add<NumaStoreOpFallback>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createNUMAOpsToLLVMPass() {
  return std::make_unique<NUMAOpsToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
