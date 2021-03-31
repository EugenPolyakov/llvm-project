#ifndef DELPHIEXTENSION_H
#define DELPHIEXTENSION_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetOptions.h>
//#include <llvm/IR/LLVMContext.h>
#ifdef _WIN32
#include <windows.h>
#undef max
#undef min
#else
#include <dlfcn.h>
#endif
#include "clang-c/Platform.h"

/*class __declspec(uuid("00000000-0000-0000-C000-000000000046")) IUnknown {
public:
  virtual HRESULT __stdcall QueryInterface(const GUID &riid,
                                           void **ppvObject) = 0;

  virtual ULONG __stdcall AddRef(void) = 0;

  virtual ULONG __stdcall Release(void) = 0;

};*/
LLVM_CLANG_C_EXTERN_C_BEGIN

using namespace llvm;

class ILLVMModule;
class IIRBuilder;

class __declspec(uuid("110F5C42-3B19-45F7-9B47-EA7C24231231")) ILLVMContext
    : public IUnknown {
  virtual ILLVMModule __stdcall *CreateModule(const char *moduleID) = 0;
  virtual IntegerType __stdcall *GetIntegerType(unsigned int numBits) = 0;
  // params + paramsHiIndex - delphi open array
  virtual FunctionType __stdcall *GetFunctionType(Type *result,
                                                        Type **params,
                                                        int paramsHiIndex,
                                                        bool isVarArg) = 0;
  virtual BasicBlock __stdcall *CreateBasicBlock(
      const char *Name = "", Function *Parent = nullptr,
      BasicBlock *InsertBefore = nullptr) = 0;
  virtual void __stdcall CreateBuilder(IIRBuilder **builder) = 0;
};

class __declspec(uuid("984F9C72-9CBE-4B35-AAAD-B677A20EFDAD")) ILLVMPassInterface
    : public IUnknown {
public:
  virtual char __stdcall *GetPassID() = 0;
};

class __declspec(uuid("179F511B-C575-45E0-ADB3-951B0AE7FA82")) ILLVMModulePass
    : public ILLVMPassInterface {
public:
  virtual bool __stdcall Run(ILLVMModule *M) = 0;
};

class ILLVMFunctionPass : public ILLVMPassInterface {};

class __declspec(uuid("5FE398C8-B062-4795-BD53-1C60C8322960")) ILLVMPassManager
    : public IUnknown {
  virtual void __stdcall AddPass(Pass *pass) = 0;
  virtual void __stdcall AddPass(ILLVMPassInterface *pass) = 0;
  virtual bool __stdcall Run(ILLVMModule *M) = 0;
};

/// A raw_ostream that writes to an ISequentialStream.  This is a simple adaptor
/// class. This class does not encounter output errors.
class Adapter_ISequentialStream : public raw_ostream {
  ISequentialStream *real;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const override { return 0; }

public:
  Adapter_ISequentialStream(ISequentialStream *stream);
  ~Adapter_ISequentialStream() override;
};

class Adapter_IStream : public raw_pwrite_stream {
  IStream *real;

  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override;
  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const override;

public:
  Adapter_IStream(IStream *stream);
  ~Adapter_IStream() override;
};

class ILLVMModule /*: public IUnknown*/ {
  virtual void __stdcall SetDataLayout(const char *dataLayout) = 0;
  virtual void __stdcall SetDataLayout(DataLayout *dataLayout) = 0;
  virtual const DataLayout __stdcall *GetDataLayout() = 0;
  virtual void __stdcall SetTargetTriple(const char *triple) = 0;
  virtual Function *__stdcall CreateFunction(
      FunctionType *Ty, GlobalValue::LinkageTypes Linkage,
      unsigned AddrSpace, const char *N) = 0;
  virtual void __stdcall WriteBitcodeToFile(ISequentialStream *out) = 0;
  virtual bool __stdcall Verify(ISequentialStream *out) = 0;
};/// Common base class shared among various IRBuilders.

class __declspec(uuid("8E6BD6AE-F0AC-4D06-A162-F8E48BF07F58")) IIRBuilder
    : public IUnknown {
public:

  /// Insert and return the specified instruction.
  virtual Instruction __stdcall *Insert(Instruction *I,
                                        const char *Name = "") = 0;

  /// No-op overload to handle constants.
  virtual Constant __stdcall *Insert(Constant *C, const char * = "") = 0;

  virtual Value __stdcall *Insert(Value *V, const char *Name = "") = 0;

  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//

  /// Clear the insertion point: created instructions will not be
  /// inserted into a block.
  virtual void __stdcall ClearInsertionPoint() = 0;

  virtual BasicBlock __stdcall *GetInsertBlock() = 0;
  //BasicBlock::iterator GetInsertPoint() const =0;
  virtual void __stdcall GetContext(ILLVMContext **context) = 0;

  /// This specifies that created instructions should be appended to the
  /// end of the specified block.
  virtual void __stdcall SetInsertPoint(BasicBlock *TheBB) = 0;

  /// This specifies that created instructions should be inserted before
  /// the specified instruction.
  virtual void __stdcall SetInsertPoint(Instruction *I) = 0;

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  /*void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) =0;*/

  /// Set location information used by debugging information.
  virtual void __stdcall SetCurrentDebugLocation(DebugLoc L) = 0;

  /// Collect metadata with IDs \p MetadataKinds from \p Src which should be
  /// added to all created instructions. Entries present in MedataDataToCopy but
  /// not on \p Src will be dropped from MetadataToCopy.
  /*void CollectMetadataToCopy(Instruction *Src,
                             ArrayRef<unsigned> MetadataKinds) =0;*/

  /// Get location information used by debugging information.
  virtual DebugLoc __stdcall getCurrentDebugLocation() = 0;

  /// If this builder has a current debug location, set it on the
  /// specified instruction.
  virtual void __stdcall SetInstDebugLocation(Instruction *I) const =0;

  /// Add all entries in MetadataToCopy to \p I.
  virtual void __stdcall AddMetadataToInst(Instruction *I) const =0;

  /// Get the return type of the current function that we're emitting
  /// into.
  virtual Type __stdcall *getCurrentFunctionReturnType() = 0;

  /// Returns the current insert point.
  virtual IRBuilderBase::InsertPoint __stdcall saveIP() = 0;

  /// Returns the current insert point, clearing it in the process.
  virtual IRBuilderBase::InsertPoint __stdcall saveAndClearIP() = 0;

  /// Sets the current insert point to a previously-saved location.
  virtual void __stdcall restoreIP(IRBuilderBase::InsertPoint IP) = 0;

  /// Get the floating point math metadata being used.
  //MDNode *getDefaultFPMathTag() const =0;

  /// Get the flags to be applied to created floating point ops
  //FastMathFlags getFastMathFlags() const =0;

  //FastMathFlags &getFastMathFlags() =0;

  /// Clear the fast-math flags.
  //void clearFastMathFlags() =0;

  /// Set the floating point math metadata to be used.
  //void setDefaultFPMathTag(MDNode *FPMathTag) =0;

  /// Set the fast-math flags to be used with generated fp-math operators
  //void setFastMathFlags(FastMathFlags NewFMF) =0;

  /// Enable/Disable use of constrained floating point math. When
  /// enabled the CreateF<op>() calls instead create constrained
  /// floating point intrinsic calls. Fast math flags are unaffected
  /// by this setting.
  //void setIsFPConstrained(bool IsCon) =0;

  /// Query for the use of constrained floating point math
  //bool getIsFPConstrained() =0;

  /// Set the exception handling to be used with constrained floating point
  /*void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) =0;*/

  /// Set the rounding mode handling to be used with constrained floating point
  /*void setDefaultConstrainedRounding(RoundingMode NewRounding) =0;*/

  /// Get the exception handling used with constrained floating point
  /*fp::ExceptionBehavior getDefaultConstrainedExcept() =0;*/

  /// Get the rounding mode handling used with constrained floating point
  /*RoundingMode getDefaultConstrainedRounding() =0;*/

  /*void setConstrainedFPFunctionAttr() =0;*/

  /*void setConstrainedFPCallAttr(CallBase *I) =0;*/

  /*void setDefaultOperandBundles(ArrayRef<OperandBundleDef> OpBundles) =0;*/

  //===--------------------------------------------------------------------===//
  // Miscellaneous creation methods.
  //===--------------------------------------------------------------------===//

  /// Make a new global variable with initializer type i8*
  ///
  /// Make a new global variable with an initializer that has array of i8 type
  /// filled in with the null terminated string value specified.  The new global
  /// variable will be marked mergable with any others of the same contents.  If
  /// Name is specified, it is the name of the global variable created.
  ///
  /// If no module is given via \p M, it is take from the insertion point basic
  /// block.
  /*GlobalVariable *CreateGlobalString(StringRef Str,
                                           const char *Name = "",
                                     unsigned AddressSpace = 0,
                                     Module *M = nullptr);*/

  /// Get a constant value representing either true or false.
  virtual ConstantInt __stdcall *getInt1(bool V) = 0;

  /// Get the constant value for i1 true.
  virtual ConstantInt __stdcall *getTrue() = 0;

  /// Get the constant value for i1 false.
  virtual ConstantInt __stdcall *getFalse() = 0;

  /// Get a constant 8-bit value.
  virtual ConstantInt __stdcall *getInt8(uint8_t C) = 0;

  /// Get a constant 16-bit value.
  virtual ConstantInt __stdcall *getInt16(uint16_t C) = 0;

  /// Get a constant 32-bit value.
  virtual ConstantInt __stdcall *getInt32(uint32_t C) = 0;

  /// Get a constant 64-bit value.
  virtual ConstantInt __stdcall *getInt64(uint64_t C) = 0;

  /// Get a constant N-bit value, zero extended or truncated from
  /// a 64-bit value.
  virtual ConstantInt __stdcall *getIntN(unsigned N, uint64_t C) = 0;

  /// Get a constant integer value.
  /*ConstantInt *getInt(const APInt &AI) =0;*/

  //===--------------------------------------------------------------------===//
  // Type creation methods
  //===--------------------------------------------------------------------===//

  /// Fetch the type representing a single bit
  virtual IntegerType __stdcall *getInt1Ty() = 0;

  /// Fetch the type representing an 8-bit integer.
  virtual IntegerType __stdcall *getInt8Ty() = 0;

  /// Fetch the type representing a 16-bit integer.
  virtual IntegerType __stdcall *getInt16Ty() = 0;

  /// Fetch the type representing a 32-bit integer.
  virtual IntegerType __stdcall *getInt32Ty() = 0;

  /// Fetch the type representing a 64-bit integer.
  virtual IntegerType __stdcall *getInt64Ty() = 0;

  /// Fetch the type representing a 128-bit integer.
  virtual IntegerType __stdcall *getInt128Ty() = 0;

  /// Fetch the type representing an N-bit integer.
  virtual IntegerType __stdcall *getIntNTy(unsigned N) = 0;

  /// Fetch the type representing a 16-bit floating point value.
  virtual Type __stdcall *getHalfTy() = 0;

  /// Fetch the type representing a 16-bit brain floating point value.
  virtual Type __stdcall *getBFloatTy() = 0;

  /// Fetch the type representing a 32-bit floating point value.
  virtual Type __stdcall *getFloatTy() = 0;

  /// Fetch the type representing a 64-bit floating point value.
  virtual Type __stdcall *getDoubleTy() = 0;

  /// Fetch the type representing void.
  virtual Type __stdcall *getVoidTy() = 0;

  /// Fetch the type representing a pointer to an 8-bit integer value.
  virtual PointerType __stdcall *getInt8PtrTy(unsigned AddrSpace = 0) = 0;

  /// Fetch the type representing a pointer to an integer value.
  virtual IntegerType __stdcall *getIntPtrTy(const DataLayout &DL,
                                                   unsigned AddrSpace = 0) = 0;

  //===--------------------------------------------------------------------===//
  // Intrinsic creation methods
  //===--------------------------------------------------------------------===//

  /// Create and insert a memset to the specified pointer and the
  /// specified value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  virtual CallInst __stdcall *
  CreateMemSet(Value *Ptr, Value *Val, uint64_t Size, MaybeAlign Align,
               bool isVolatile = false, MDNode *TBAATag = nullptr,
               MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateMemSet(Value *Ptr, Value *Val, Value *Size, MaybeAlign Align,
               bool isVolatile = false, MDNode *TBAATag = nullptr,
               MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  /// Create and insert an element unordered-atomic memset of the region of
  /// memory starting at the given pointer to the given value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  virtual CallInst __stdcall *CreateElementUnorderedAtomicMemSet(
      Value *Ptr, Value *Val, uint64_t Size, Align Alignment,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *CreateElementUnorderedAtomicMemSet(
      Value *Ptr, Value *Val, Value *Size, Align Alignment,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  /// Create and insert a memcpy between the specified pointers.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  virtual CallInst __stdcall *
  CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src, MaybeAlign SrcAlign,
               uint64_t Size,
               bool isVolatile = false, MDNode *TBAATag = nullptr,
               MDNode *TBAAStructTag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *CreateMemTransferInst(
      Intrinsic::ID IntrID, Value *Dst, MaybeAlign DstAlign, Value *Src,
      MaybeAlign SrcAlign, Value *Size,
      bool isVolatile = false, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src, MaybeAlign SrcAlign,
               Value *Size,
               bool isVolatile = false, MDNode *TBAATag = nullptr,
               MDNode *TBAAStructTag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *CreateMemCpyInline(Value *Dst,
                                                 MaybeAlign DstAlign, Value *Src,
                     MaybeAlign SrcAlign,
                     Value *Size) = 0;

  /// Create and insert an element unordered-atomic memcpy between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers,
  /// respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  virtual CallInst __stdcall *CreateElementUnorderedAtomicMemCpy(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign,
      Value *Size, uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                MaybeAlign SrcAlign, uint64_t Size,
                bool isVolatile = false, MDNode *TBAATag = nullptr,
                MDNode *ScopeTag = nullptr,
                MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                MaybeAlign SrcAlign, Value *Size, bool isVolatile = false,
                MDNode *TBAATag = nullptr,
                MDNode *ScopeTag = nullptr,
                MDNode *NoAliasTag = nullptr) = 0;

  /// \brief Create and insert an element unordered-atomic memmove between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers,
  /// respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  virtual CallInst __stdcall *CreateElementUnorderedAtomicMemMove(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign,
      uint64_t Size, uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) = 0;

  virtual CallInst __stdcall *CreateElementUnorderedAtomicMemMove(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign,
      Value *Size, uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) = 0;

  /// Create a vector fadd reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value for ordered reductions.
  virtual CallInst __stdcall *CreateFAddReduce(Value *Acc,
                                                     Value *Src) = 0;

  /// Create a vector fmul reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value for ordered reductions.
  virtual CallInst __stdcall *CreateFMulReduce(Value *Acc,
                                                     Value *Src) = 0;

  /// Create a vector int add reduction intrinsic of the source vector.
  virtual CallInst __stdcall *CreateAddReduce(Value *Src) = 0;

  /// Create a vector int mul reduction intrinsic of the source vector.
  virtual CallInst __stdcall *CreateMulReduce(Value *Src) = 0;

  /// Create a vector int AND reduction intrinsic of the source vector.
  virtual CallInst __stdcall *CreateAndReduce(Value *Src) = 0;

  /// Create a vector int OR reduction intrinsic of the source vector.
  virtual CallInst __stdcall *CreateOrReduce(Value *Src) = 0;

  /// Create a vector int XOR reduction intrinsic of the source vector.
  virtual CallInst __stdcall *CreateXorReduce(Value *Src) = 0;

  /// Create a vector integer max reduction intrinsic of the source
  /// vector.
  virtual CallInst __stdcall *
  CreateIntMaxReduce(Value *Src, bool IsSigned = false) = 0;

  /// Create a vector integer min reduction intrinsic of the source
  /// vector.
  virtual CallInst __stdcall *
  CreateIntMinReduce(Value *Src, bool IsSigned = false) = 0;

  /// Create a vector float max reduction intrinsic of the source
  /// vector.
  virtual CallInst __stdcall *CreateFPMaxReduce(Value *Src) = 0;

  /// Create a vector float min reduction intrinsic of the source
  /// vector.
  virtual CallInst __stdcall *CreateFPMinReduce(Value *Src) = 0;

  /// Create a lifetime.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  virtual CallInst __stdcall *
  CreateLifetimeStart(Value *Ptr, ConstantInt *Size = nullptr) = 0;

  /// Create a lifetime.end intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  virtual CallInst __stdcall *
  CreateLifetimeEnd(Value *Ptr, ConstantInt *Size = nullptr) = 0;

  /// Create a call to invariant.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  virtual CallInst __stdcall *
  CreateInvariantStart(Value *Ptr, ConstantInt *Size = nullptr) = 0;

  /// Create a call to Masked Load intrinsic
  virtual CallInst __stdcall *CreateMaskedLoad(Value *Ptr, Align Alignment,
                                               Value *Mask,
                                               Value *PassThru = nullptr,
                                               const char *Name = "") = 0;

  /// Create a call to Masked Store intrinsic
  virtual CallInst __stdcall *
  CreateMaskedStore(Value *Val, Value *Ptr, Align Alignment, Value *Mask) = 0;

  /// Create a call to Masked Gather intrinsic
  virtual CallInst __stdcall *CreateMaskedGather(Value *Ptrs, Align Alignment,
                                                 Value *Mask = nullptr,
                                                 Value *PassThru = nullptr,
                                                 const char *Name = "") = 0;

  /// Create a call to Masked Scatter intrinsic
  virtual CallInst __stdcall *CreateMaskedScatter(Value *Val, Value *Ptrs,
                                                  Align Alignment,
                      Value *Mask = nullptr) = 0;

  /// Create an assume intrinsic call that allows the optimizer to
  /// assume that the provided condition will be true.
  ///
  /// The optional argument \p OpBundles specifies operand bundles that are
  /// added to the call instruction.
  /*virtual CallInst __stdcall *
  CreateAssumption(Value *Cond,
                   ArrayRef<OperandBundleDef> OpBundles = None) = 0;*/

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  virtual CallInst __stdcall *
  CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                         Value *ActualCallee, Value **CallArgs,
                         int CallArgsHiIndex, Value **DeoptArgs,
                         int DeoptArgsHiIndex, Value **GCArgs,
                         int GCArgsHiIndex, const char *Name = "") = 0;

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  virtual CallInst __stdcall *CreateGCStatepointCall(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualCallee, uint32_t Flags,
      Value **CallArgs, int CallArgsHiIndex, Use *TransitionArgs,
      int TransitionArgsHiIndex, Use *DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") = 0;

  /// Conveninence function for the common case when CallArgs are filled
  /// in using makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be
  /// .get()'ed to get the Value pointer.
  virtual CallInst __stdcall *
  CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                         Value *ActualCallee, Use* CallArgs,
                         int CallArgsHiIndex,
      Value ** DeoptArgs, int DeoptArgsHiIndex,
      Value ** GCArgs, int GCArgsHiIndex, const char *Name = "") = 0;

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  virtual InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, Value **InvokeArgs,
      int InvokeArgsHiIndex, Value **DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") = 0;

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  virtual InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, uint32_t Flags,
      Value **InvokeArgs, int InvokeArgsHiIndex, Use *TransitionArgs,
      int TransitionArgsHiIndex, Use *DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") = 0;

  // Convenience function for the common case when CallArgs are filled in using
  // makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be .get()'ed to
  // get the Value *.
  virtual InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, Use *InvokeArgs,
      int InvokeArgsHiIndex, Value **DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") = 0;

  /// Create a call to the experimental.gc.result intrinsic to extract
  /// the result from a call wrapped in a statepoint.
  virtual CallInst __stdcall *
  CreateGCResult(Instruction *Statepoint, Type *ResultType,
                                                   const char *Name = "") = 0;

  /// Create a call to the experimental.gc.relocate intrinsics to
  /// project the relocated value of one pointer from the statepoint.
  virtual CallInst __stdcall *
  CreateGCRelocate(Instruction *Statepoint,
                                           int BaseOffset,
                   int DerivedOffset, Type *ResultType,
                   const char *Name = "") = 0;

  /// Create a call to llvm.vscale, multiplied by \p Scaling. The type of VScale
  /// will be the same type as that of \p Scaling.
  virtual Value __stdcall *CreateVScale(Constant *Scaling,
                                              const char *Name = "") = 0;

  /// Create a call to intrinsic \p ID with 1 operand which is mangled on its
  /// type.
  virtual CallInst __stdcall *
  CreateUnaryIntrinsic(Intrinsic::ID ID, Value *V,
                       Instruction *FMFSource = nullptr,
                       const char *Name = "") = 0;

  /// Create a call to intrinsic \p ID with 2 operands which is mangled on the
  /// first type.
  virtual CallInst __stdcall *
  CreateBinaryIntrinsic(Intrinsic::ID ID, Value *LHS, Value *RHS,
                        Instruction *FMFSource = nullptr,
                        const char *Name = "") = 0;

  /// Create a call to intrinsic \p ID with \p args, mangled using \p Types. If
  /// \p FMFSource is provided, copy fast-math-flags from that instruction to
  /// the intrinsic.
  virtual CallInst __stdcall *
  CreateIntrinsic(Intrinsic::ID ID, Type ** Types, int TypesHiIndex,
                  Value ** Args,
                  int ArgsHiIndex,
      Instruction *FMFSource = nullptr,
                  const char *Name = "") = 0;

  /// Create call to the minnum intrinsic.
  virtual CallInst __stdcall *
  CreateMinNum(Value *LHS, Value *RHS,
                                                 const char *Name = "") = 0;

  /// Create call to the maxnum intrinsic.
  virtual CallInst __stdcall *
  CreateMaxNum(Value *LHS, Value *RHS,
                                                 const char *Name = "") = 0;

  /// Create call to the minimum intrinsic.
  virtual CallInst __stdcall *
  CreateMinimum(Value *LHS, Value *RHS,
                                                  const char *Name = "") = 0;

  /// Create call to the maximum intrinsic.
  virtual CallInst __stdcall *
  CreateMaximum(Value *LHS, Value *RHS,
                                                  const char *Name = "") = 0;

  /// Create a call to the experimental.vector.extract intrinsic.
  virtual CallInst __stdcall *
  CreateExtractVector(Type *DstType, Value *SrcVec,
                      Value *Idx,
                      const char *Name = "") = 0;

  /// Create a call to the experimental.vector.insert intrinsic.
  virtual CallInst __stdcall *
  CreateInsertVector(Type *DstType, Value *SrcVec,
                     Value *SubVec, Value *Idx,
                     const char *Name = "") = 0;

  /// Create a 'ret void' instruction.
  virtual ReturnInst __stdcall *CreateRetVoid() = 0;

  /// Create a 'ret <val>' instruction.
  virtual ReturnInst __stdcall *CreateRet(Value *V) = 0;

  /// Create a sequence of N insertvalue instructions,
  /// with one Value from the retVals array each, that build a aggregate
  /// return value one value at a time, and a ret instruction to return
  /// the resulting aggregate value.
  ///
  /// This is a convenience function for code that uses aggregate return values
  /// as a vehicle for having multiple return values.
  virtual ReturnInst __stdcall *CreateAggregateRet(Value *const *retVals,
                                                   int retValsHiIndex) = 0;

  /// Create an unconditional 'br label X' instruction.
  virtual BranchInst __stdcall *CreateBr(BasicBlock *Dest) = 0;

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  virtual BranchInst __stdcall *
  CreateCondBr(Value *Cond, BasicBlock *True,
               BasicBlock *False,
                           MDNode *BranchWeights = nullptr,
               MDNode *Unpredictable = nullptr) = 0;

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction. Copy branch meta data if available.
  virtual BranchInst __stdcall *CreateCondBr(Value *Cond,
                                                   BasicBlock *True,
                                                   BasicBlock *False, Instruction *MDSrc) = 0;

  /// Create a switch instruction with the specified value, default dest,
  /// and with a hint for the number of cases that will be added (for efficient
  /// allocation).
  virtual SwitchInst __stdcall *
  CreateSwitch(Value *V, BasicBlock *Dest,
                                         unsigned NumCases = 10,
               MDNode *BranchWeights = nullptr,
               MDNode *Unpredictable = nullptr) = 0;

  /// Create an indirect branch instruction with the specified address
  /// operand, with an optional hint for the number of destinations that will be
  /// added (for efficient allocation).
  virtual IndirectBrInst __stdcall *CreateIndirectBr(Value *Addr,
                                                 unsigned NumDests = 10) = 0;

  /// Create an invoke instruction.
  virtual InvokeInst __stdcall *
  CreateInvoke(FunctionType *Ty, Value *Callee,
                           BasicBlock *NormalDest, BasicBlock *UnwindDest,
                           Value ** Args, int ArgsHiIndex,
               OperandBundleDef* OpBundles, int OpBundlesHiIndex,
               const char *Name = "") = 0;

  virtual InvokeInst __stdcall *CreateInvoke(FunctionType *Ty, Value *Callee,
                                             BasicBlock *NormalDest,
                                             BasicBlock *UnwindDest,
                                             Value **Args, int ArgsHiIndex,
                                             const char *Name = "") = 0;

  virtual InvokeInst __stdcall *
  CreateInvoke(FunctionCallee Callee, BasicBlock *NormalDest,
               BasicBlock *UnwindDest, Value **Args, int ArgsHiIndex,
               OperandBundleDef *OpBundles, int OpBundlesHiIndex,
               const char *Name = "") = 0;

  virtual InvokeInst __stdcall *CreateInvoke(FunctionCallee Callee,
                                             BasicBlock *NormalDest,
                                             BasicBlock *UnwindDest,
                                             Value **Args, int ArgsHiIndex,
                                             const char *Name = "") = 0;

  /// \brief Create a callbr instruction.
  virtual CallBrInst __stdcall *
  CreateCallBr(FunctionType *Ty, Value *Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, const char *Name = "") = 0;

  virtual CallBrInst __stdcall *
  CreateCallBr(FunctionType *Ty, Value *Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, OperandBundleDef *OpBundles,
               int OpBundlesHiIndex, const char *Name = "") = 0;

  virtual CallBrInst __stdcall *
  CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, const char *Name = "") = 0;

  virtual CallBrInst __stdcall *
  CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, OperandBundleDef *OpBundles,
               int OpBundlesHiIndex, const char *Name = "") = 0;

  virtual ResumeInst __stdcall *CreateResume(Value *Exn) = 0;

  virtual CleanupReturnInst __stdcall *
  CreateCleanupRet(CleanupPadInst *CleanupPad,
                   BasicBlock *UnwindBB = nullptr) = 0;

  virtual CatchSwitchInst __stdcall *
  CreateCatchSwitch(Value *ParentPad,
                                                   BasicBlock *UnwindBB,
                                     unsigned NumHandlers, const char *Name = "") = 0;

  virtual CatchPadInst __stdcall *CreateCatchPad(Value *ParentPad, Value **Args,
                                                 int ArgsHiIndex,
                                                 const char *Name = "") = 0;

  virtual CleanupPadInst __stdcall *CreateCleanupPad(Value *ParentPad,
                                                     Value **Args,
                                                     int ArgsHiIndex,
                                                     const char *Name = "") = 0;

  virtual CatchReturnInst __stdcall *
  CreateCatchRet(CatchPadInst *CatchPad, BasicBlock *BB) = 0;

  virtual UnreachableInst __stdcall *CreateUnreachable() = 0;
  //has auto simplification and can const + const convert to const
  virtual Value __stdcall *CreateAdd(Value *LHS, Value *RHS,
                                           const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  virtual Value __stdcall *CreateNSWAdd(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateNUWAdd(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateSub(Value *LHS, Value *RHS,
                                           const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  virtual Value __stdcall *CreateNSWSub(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateNUWSub(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateMul(Value *LHS, Value *RHS,
                                           const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  virtual Value __stdcall *CreateNSWMul(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateNUWMul(Value *LHS, Value *RHS,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateUDiv(Value *LHS, Value *RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  virtual Value __stdcall *CreateExactUDiv(Value *LHS, Value *RHS,
                                                 const char *Name = "") = 0;

  virtual Value __stdcall *CreateSDiv(Value *LHS, Value *RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  virtual Value __stdcall *CreateExactSDiv(Value *LHS, Value *RHS,
                                                 const char *Name = "") = 0;

  virtual Value __stdcall *CreateURem(Value *LHS, Value *RHS,
                                            const char *Name = "") = 0;

  virtual Value __stdcall *CreateSRem(Value *LHS, Value *RHS,
                                            const char *Name = "") = 0;

  virtual Value __stdcall *CreateShl(Value *LHS, Value *RHS,
                                           const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  /*virtual Value __stdcall *CreateShl(Value *LHS, const APInt &RHS,
                                 const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;*/

  virtual Value __stdcall *CreateShl(Value *LHS, uint64_t RHS,
                                 const char *Name = "",
                                           bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  virtual Value __stdcall *CreateLShr(Value *LHS, Value *RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  /*virtual Value __stdcall *CreateLShr(Value *LHS, const APInt &RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;*/

  virtual Value __stdcall *CreateLShr(Value *LHS, uint64_t RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  virtual Value __stdcall *CreateAShr(Value *LHS, Value *RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  /*virtual Value __stdcall *CreateAShr(Value *LHS, const APInt &RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;*/

  virtual Value __stdcall *CreateAShr(Value *LHS, uint64_t RHS,
                                  const char *Name = "",
                                            bool isExact = false) = 0;

  virtual Value __stdcall *CreateAnd(Value *LHS, Value *RHS,
                                           const char *Name = "") = 0;

  /*virtual Value __stdcall *CreateAnd(Value *LHS, const APInt &RHS,
                                           const char *Name = "") = 0;*/

  virtual Value __stdcall *CreateAnd(Value *LHS, uint64_t RHS,
                                           const char *Name = "") = 0;

  virtual Value __stdcall *CreateAnd(Value ** Ops, int OpsHiIndex) = 0;

  virtual Value __stdcall *CreateOr(Value *LHS, Value *RHS,
                                          const char *Name = "") = 0;

  /*virtual Value __stdcall *CreateOr(Value *LHS, const APInt &RHS,
                                          const char *Name = "") = 0;*/

  virtual Value __stdcall *CreateOr(Value *LHS, uint64_t RHS,
                                          const char *Name = "") = 0;

  virtual Value __stdcall *CreateOr(Value **Ops, int OpsHiIndex) = 0;

  virtual Value __stdcall *CreateXor(Value *LHS, Value *RHS,
                                           const char *Name = "") = 0;

  /*virtual Value __stdcall *CreateXor(Value *LHS, const APInt &RHS,
                                           const char *Name = "") = 0;*/

  virtual Value __stdcall *CreateXor(Value *LHS, uint64_t RHS,
                                           const char *Name = "") = 0;

  virtual Value __stdcall *CreateFAdd(Value *L, Value *R,
                                            const char *Name = "",
                                            MDNode *FPMD = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFAddFMF(Value *L, Value *R,
                                               Instruction *FMFSource,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateFSub(Value *L, Value *R,
                                            const char *Name = "",
                                            MDNode *FPMD = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFSubFMF(Value *L, Value *R,
                                               Instruction *FMFSource,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateFMul(Value *L, Value *R,
                                            const char *Name = "",
                                            MDNode *FPMD = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFMulFMF(Value *L, Value *R,
                                               Instruction *FMFSource,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateFDiv(Value *L, Value *R,
                                            const char *Name = "",
                                            MDNode *FPMD = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFDivFMF(Value *L, Value *R,
                                               Instruction *FMFSource,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateFRem(Value *L, Value *R, const char *Name = "",
                                      MDNode *FPMD = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFRemFMF(Value *L, Value *R,
                                         Instruction *FMFSource,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateBinOp(Instruction::BinaryOps Opc, Value *LHS,
                                       Value *RHS, const char *Name = "",
                                       MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateConstrainedFPBinOp(Intrinsic::ID ID, Value *L, Value *R,
                           Instruction *FMFSource = nullptr,
                           const char *Name = "", MDNode *FPMathTag = nullptr,
                           Optional<RoundingMode> Rounding = None,
                           Optional<fp::ExceptionBehavior> Except = None) = 0;

  virtual Value __stdcall *CreateNeg(Value *V, const char *Name = "",
                                 bool HasNUW = false,
                                           bool HasNSW = false) = 0;

  virtual Value __stdcall *CreateNSWNeg(Value *V,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateNUWNeg(Value *V,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateFNeg(Value *V, const char *Name = "",
                                            MDNode *FPMathTag = nullptr) = 0;

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  virtual Value __stdcall *CreateFNegFMF(Value *V, Instruction *FMFSource,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateNot(Value *V, const char *Name = "") = 0;

  virtual Value __stdcall *CreateUnOp(Instruction::UnaryOps Opc, Value *V,
                                      const char *Name = "",
                                      MDNode *FPMathTag = nullptr) = 0;

  /// Create either a UnaryOperator or BinaryOperator depending on \p Opc.
  /// Correct number of operands must be passed accordingly.
  virtual Value __stdcall *CreateNAryOp(unsigned Opc, Value **Ops,
                                        int OpsHiIndex, const char *Name = "",
                                        MDNode *FPMathTag = nullptr) = 0;

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  virtual AllocaInst __stdcall *CreateAlloca(Type *Ty, unsigned AddrSpace,
                                             Value *ArraySize = nullptr,
                                             const char *Name = "") = 0;

  virtual AllocaInst __stdcall *
  CreateAlloca(Type *Ty, Value *ArraySize = nullptr, const char *Name = "") = 0;

  virtual LoadInst __stdcall *CreateLoad(Type *Ty, Value *Ptr, bool isVolatile,
                                         MaybeAlign Align,
                                         const char *Name = "") = 0;

  virtual StoreInst __stdcall *CreateStore(Value *Val, Value *Ptr,
                                           MaybeAlign Align,
                                           bool isVolatile = false) = 0;

  virtual FenceInst __stdcall *
  CreateFence(AtomicOrdering Ordering, SyncScope::ID SSID = SyncScope::System,
              const char *Name = "") = 0;

  virtual AtomicCmpXchgInst __stdcall *
  CreateAtomicCmpXchg(Value *Ptr, Value *Cmp, Value *New,
                      AtomicOrdering SuccessOrdering,
                      AtomicOrdering FailureOrdering,
                      SyncScope::ID SSID = SyncScope::System) = 0;

  virtual AtomicRMWInst __stdcall *
  CreateAtomicRMW(AtomicRMWInst::BinOp Op, Value *Ptr, Value *Val,
                  AtomicOrdering Ordering,
                  SyncScope::ID SSID = SyncScope::System) = 0;

  virtual Value __stdcall *CreateGEP(Value *Ptr, Value **IdxList,
                                     int IdxListHiIndex,
                                     const char *Name = "") = 0;

  virtual Value __stdcall *CreateGEP(Type *Ty, Value *Ptr, Value **IdxList,
                                     int IdxListHiIndex,
                                     const char *Name = "") = 0;

  virtual Value __stdcall *CreateInBoundsGEP(Value *Ptr, Value **IdxList,
                                             int IdxListHiIndex,
                                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateInBoundsGEP(Type *Ty, Value *Ptr,
                                             Value **IdxList,
                                             int IdxListHiIndex,
                                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateGEP(Value *Ptr, Value *Idx,
                                     const char *Name = "") = 0;

  virtual Value __stdcall *CreateGEP(Type *Ty, Value *Ptr, Value *Idx,
                                     const char *Name = "") = 0;

  virtual Value __stdcall *CreateInBoundsGEP(Type *Ty, Value *Ptr, Value *Idx,
                                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateConstGEP1_32(Type *Ty, Value *Ptr,
                                              unsigned Idx0,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *
  CreateConstInBoundsGEP1_32(Type *Ty, Value *Ptr, unsigned Idx0,
                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateConstGEP2_32(Type *Ty, Value *Ptr,
                                              unsigned Idx0, unsigned Idx1,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *
  CreateConstInBoundsGEP2_32(Type *Ty, Value *Ptr, unsigned Idx0, unsigned Idx1,
                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateConstGEP1_64(Type *Ty, Value *Ptr,
                                              uint64_t Idx0,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *
  CreateConstInBoundsGEP1_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                             const char *Name = "") = 0;

  virtual Value __stdcall *CreateConstGEP2_64(Type *Ty, Value *Ptr,
                                              uint64_t Idx0, uint64_t Idx1,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *
  CreateConstInBoundsGEP2_64(Type *Ty, Value *Ptr, uint64_t Idx0, uint64_t Idx1,
                             const char *Name = "") = 0;

  virtual Value __stdcall *
  CreateStructGEP(Type *Ty, Value *Ptr, unsigned Idx,
                                           const char *Name = "") = 0;

  /// Same as CreateGlobalString, but return a pointer with "i8*" type
  /// instead of a pointer to array of i8.
  ///
  /// If no module is given via \p M, it is take from the insertion point basic
  /// block.
  /*virtual Constant __stdcall *CreateGlobalStringPtr(StringRef Str,
                                                    const char *Name = "",
                                                    unsigned AddressSpace = 0,
                                                    Module *M = nullptr) = 0;*/

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  virtual Value __stdcall *CreateTrunc(Value *V, Type *DestTy,
                                       const char *Name = "") = 0;

  virtual Value __stdcall *CreateZExt(Value *V, Type *DestTy,
                                      const char *Name = "") = 0;

  virtual Value __stdcall *CreateSExt(Value *V, Type *DestTy,
                                      const char *Name = "") = 0;

  /// Create a ZExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  virtual Value __stdcall *CreateZExtOrTrunc(Value *V, Type *DestTy,
                                             const char *Name = "");

  /// Create a SExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  virtual Value __stdcall *CreateSExtOrTrunc(Value *V, Type *DestTy,
                                             const char *Name = "");

  virtual Value __stdcall *CreateFPToUI(Value *V, Type *DestTy,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateFPToSI(Value *V, Type *DestTy,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateUIToFP(Value *V, Type *DestTy,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateSIToFP(Value *V, Type *DestTy,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateFPTrunc(Value *V, Type *DestTy,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateFPExt(Value *V, Type *DestTy,
                                       const char *Name = "") = 0;

  virtual Value __stdcall *CreatePtrToInt(Value *V, Type *DestTy,
                                          const char *Name = "") = 0;

  virtual Value __stdcall *CreateIntToPtr(Value *V, Type *DestTy,
                                          const char *Name = "") = 0;

  virtual Value __stdcall *CreateBitCast(Value *V, Type *DestTy,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateAddrSpaceCast(Value *V, Type *DestTy,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateZExtOrBitCast(Value *V, Type *DestTy,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateSExtOrBitCast(Value *V, Type *DestTy,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateTruncOrBitCast(Value *V, Type *DestTy,
                                                const char *Name = "") = 0;

  virtual Value __stdcall *CreateCast(Instruction::CastOps Op, Value *V,
                                      Type *DestTy, const char *Name = "") = 0;

  virtual Value __stdcall *CreatePointerCast(Value *V, Type *DestTy,
                                             const char *Name = "") = 0;

  virtual Value __stdcall *
  CreatePointerBitCastOrAddrSpaceCast(Value *V, Type *DestTy,
                                      const char *Name = "") = 0;

  virtual Value __stdcall *CreateIntCast(Value *V, Type *DestTy, bool isSigned,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateBitOrPointerCast(Value *V, Type *DestTy,
                                                  const char *Name = "") = 0;

  virtual Value __stdcall *CreateFPCast(Value *V, Type *DestTy,
                                        const char *Name = "") = 0;

  virtual CallInst __stdcall *
  CreateConstrainedFPCast(Intrinsic::ID ID, Value *V, Type *DestTy,
                          Instruction *FMFSource = nullptr,
                          const char *Name = "", MDNode *FPMathTag = nullptr,
                          Optional<RoundingMode> Rounding = None,
                          Optional<fp::ExceptionBehavior> Except = None) = 0;

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//

  virtual Value __stdcall *CreateICmpEQ(Value *LHS, Value *RHS,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpNE(Value *LHS, Value *RHS,
                                        const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpUGT(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpUGE(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpULT(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpULE(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpSGT(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpSGE(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpSLT(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateICmpSLE(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  virtual Value __stdcall *CreateFCmpOEQ(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpOGT(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpOGE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpOLT(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpOLE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpONE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpORD(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpUNO(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpUEQ(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpUGT(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpUGE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpULT(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpULE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateFCmpUNE(Value *LHS, Value *RHS,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateICmp(CmpInst::Predicate P, Value *LHS,
                                      Value *RHS, const char *Name = "") = 0;

  // Create a quiet floating-point comparison (i.e. one that raises an FP
  // exception only in the case where an input is a signaling NaN).
  // Note that this differs from CreateFCmpS only if IsFPConstrained is true.
  virtual Value __stdcall *CreateFCmp(CmpInst::Predicate P, Value *LHS,
                                      Value *RHS, const char *Name = "",
                                      MDNode *FPMathTag = nullptr) = 0;

  virtual Value __stdcall *CreateCmp(CmpInst::Predicate Pred, Value *LHS,
                                     Value *RHS, const char *Name = "",
                                     MDNode *FPMathTag = nullptr) = 0;

  // Create a signaling floating-point comparison (i.e. one that raises an FP
  // exception whenever an input is any NaN, signaling or quiet).
  // Note that this differs from CreateFCmp only if IsFPConstrained is true.
  virtual Value __stdcall *CreateFCmpS(CmpInst::Predicate P, Value *LHS,
                                       Value *RHS, const char *Name = "",
                                       MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateConstrainedFPCmp(Intrinsic::ID ID, CmpInst::Predicate P, Value *L,
                         Value *R, const char *Name = "",
                         Optional<fp::ExceptionBehavior> Except = None) = 0;

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  virtual PHINode __stdcall *CreatePHI(Type *Ty, unsigned NumReservedValues,
                                       const char *Name = "") = 0;

  virtual CallInst __stdcall *CreateCall(FunctionType *FTy, Value *Callee,
                                         Value ** Args, int ArgsHiIndex,
                                         const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateCall(FunctionType *FTy, Value *Callee, Value **Args, int ArgsHiIndex,
             OperandBundleDef *OpBundles, int OpBundlesHiIndex,
             const char *Name = "", MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *CreateCall(FunctionCallee Callee, Value **Args,
                                         int ArgsHiIndex, const char *Name = "",
                                         MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateCall(FunctionCallee Callee, Value **Args, int ArgsHiIndex,
             OperandBundleDef *OpBundles, int OpBundlesHiIndex,
             const char *Name = "", MDNode *FPMathTag = nullptr) = 0;

  virtual CallInst __stdcall *
  CreateConstrainedFPCall(Function *Callee, Value **Args, int ArgsHiIndex,
                          const char *Name = "",
                          Optional<RoundingMode> Rounding = None,
                          Optional<fp::ExceptionBehavior> Except = None) = 0;

  virtual Value __stdcall *CreateSelect(Value *C, Value *True, Value *False,
                                        const char *Name = "",
                                        Instruction *MDFrom = nullptr) = 0;

  virtual VAArgInst __stdcall *CreateVAArg(Value *List, Type *Ty,
                                           const char *Name = "") = 0;

  virtual Value __stdcall *CreateExtractElement(Value *Vec, Value *Idx,
                                                const char *Name = "") = 0;

  virtual Value __stdcall *CreateExtractElement(Value *Vec, uint64_t Idx,
                                                const char *Name = "") = 0;

  virtual Value __stdcall *CreateInsertElement(Value *Vec, Value *NewElt,
                                               Value *Idx,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateInsertElement(Value *Vec, Value *NewElt,
                                               uint64_t Idx,
                                               const char *Name = "") = 0;

  /*virtual Value __stdcall *CreateShuffleVector(Value *V1, Value *V2,
                                                     Value *Mask,
                                               const char *Name = "") = 0;*/

  /// See class ShuffleVectorInst for a description of the mask representation.
  virtual Value __stdcall *CreateShuffleVector(Value *V1, Value *V2, int *Mask,
                                               int MaxHiIndex,
                                               const char *Name = "") = 0;

  /// Create a unary shuffle. The second vector operand of the IR instruction
  /// is poison.
  virtual Value __stdcall *CreateShuffleVector(Value *V, int *Mask,
                                               int MaxHiIndex,
                                               const char *Name = "") = 0;

  virtual Value __stdcall *CreateExtractValue(Value *Agg, unsigned *Idxs,
                                              int IdxsHiIndex,
                                              const char *Name = "") = 0;

  virtual Value __stdcall *CreateInsertValue(Value *Agg, Value *Val,
                                             unsigned *Idxs, int IdxsHiIndex,
                                             const char *Name = "") = 0;

  virtual LandingPadInst __stdcall *
  CreateLandingPad(Type *Ty, unsigned NumClauses, const char *Name = "") = 0;

  virtual Value __stdcall *CreateFreeze(Value *V, const char *Name = "") = 0;

  //===--------------------------------------------------------------------===//
  // Utility creation methods
  //===--------------------------------------------------------------------===//

  /// Return an i1 value testing if \p Arg is null.
  virtual Value __stdcall *CreateIsNull(Value *Arg, const char *Name = "") = 0;

  /// Return an i1 value testing if \p Arg is not null.
  virtual Value __stdcall *CreateIsNotNull(Value *Arg,
                                           const char *Name = "") = 0;

  /// Return the i64 difference between two pointer values, dividing out
  /// the size of the pointed-to objects.
  ///
  /// This is intended to implement C-style pointer subtraction. As such, the
  /// pointers must be appropriately aligned for their element types and
  /// pointing into the same object.
  virtual Value __stdcall *CreatePtrDiff(Value *LHS, Value *RHS,
                                         const char *Name = "") = 0;

  /// Create a launder.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  virtual Value __stdcall *CreateLaunderInvariantGroup(Value *Ptr) = 0;

  /// \brief Create a strip.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  virtual Value __stdcall *CreateStripInvariantGroup(Value *Ptr) = 0;

  /// Return a vector value that contains \arg V broadcasted to \p
  /// NumElts elements.
  virtual Value __stdcall *CreateVectorSplat(unsigned NumElts, Value *V,
                                             const char *Name = "") = 0;

  /// Return a vector value that contains \arg V broadcasted to \p
  /// EC elements.
  virtual Value __stdcall *CreateVectorSplat(ElementCount EC, Value *V,
                                             const char *Name = "") = 0;

  /// Return a value that has been extracted from a larger integer type.
  virtual Value __stdcall *CreateExtractInteger(const DataLayout &DL,
                                                Value *From,
                                                IntegerType *ExtractedTy,
                                                uint64_t Offset,
                                                const char *Name) = 0;

  virtual Value __stdcall *
  CreatePreserveArrayAccessIndex(Type *ElTy, Value *Base, unsigned Dimension,
                                 unsigned LastIndex, MDNode *DbgInfo) = 0;

  virtual Value __stdcall *CreatePreserveUnionAccessIndex(Value *Base,
                                                          unsigned FieldIndex,
                                                          MDNode *DbgInfo) = 0;

  virtual Value __stdcall *
  CreatePreserveStructAccessIndex(Type *ElTy, Value *Base, unsigned Index,
                                  unsigned FieldIndex, MDNode *DbgInfo) = 0;

  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  virtual CallInst __stdcall *
  CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                            unsigned Alignment,
                            Value *OffsetValue = nullptr) = 0;

  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  ///
  /// This overload handles the condition where the Alignment is dependent
  /// on an existing value rather than a static value.
  virtual CallInst __stdcall *
  CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                            Value *Alignment, Value *OffsetValue = nullptr) = 0;
};

CINDEX_LINKAGE void createPassManager(ILLVMPassManager **ppvContext);
CINDEX_LINKAGE void createContext(ILLVMContext **ppvContext);

//FunctionType

//IntegerType

//Function
CINDEX_LINKAGE Function *
Function_Create(FunctionType *Ty, GlobalValue::LinkageTypes Linkage,
                unsigned AddrSpace, const char *N);
CINDEX_LINKAGE void FunctionSetCallingConversion(Function *func,
                                                 CallingConv::ID CC);
CINDEX_LINKAGE Argument *FunctionGetArgument(Function *func,
                                                   unsigned int index);
CINDEX_LINKAGE unsigned int FunctionArgumentCount(Function *func);
CINDEX_LINKAGE BasicBlock *
FunctionCreateBasicBlock(Function *parent, const char *name = "",
                         BasicBlock *insertBefore = nullptr);
CINDEX_LINKAGE bool VerifyFunction(const Function *F, ISequentialStream *out);

//Value
CINDEX_LINKAGE void ValueSetName(Value *arg, const char *name);
CINDEX_LINKAGE size_t ValueGetName(Value *arg, char *buffer, size_t bufferSize);
//Argument : Value

//AllocaInst : Instruction
CINDEX_LINKAGE AllocaInst *
AllocaInst_CreateBefore(Type *type, unsigned addrSpace, Value *arraySize,
                        Align align, const char *name,
                        Instruction *insertBefore);
CINDEX_LINKAGE AllocaInst *
AllocaInst_CreateDefaultBefore(Type *type, Value *arraySize,
                               const char *name,
                               Instruction *insertBefore);
CINDEX_LINKAGE AllocaInst *
AllocaInst_CreateAtEnd(Type *type, unsigned addrSpace, Value *arraySize,
                       Align align, const char *name, BasicBlock *insertAtEnd);
CINDEX_LINKAGE AllocaInst *
AllocaInst_CreateDefaultAtEnd(Type *type, Value *arraySize, const char *name,
                              BasicBlock *insertAtEnd);
CINDEX_LINKAGE void AllocaInstSetAlignment(AllocaInst *inst, Align align);

//StoreInst : Instruction
CINDEX_LINKAGE StoreInst *StoreInst_CreateBefore(Value *Val, Value *Ptr,
                                                 bool isVolatile, Align align,
                                                 AtomicOrdering Order,
                                                 SyncScope::ID SSID,
                                                 Instruction *InsertBefore);

CINDEX_LINKAGE StoreInst *
StoreInst_CreateDefaultBefore(Value *Val, Value *Ptr,
                              bool isVolatile, Instruction *InsertBefore);

CINDEX_LINKAGE StoreInst *StoreInst_CreateAtEnd(Value *Val, Value *Ptr,
                                                bool isVolatile, Align align,
                                                AtomicOrdering Order,
                                                SyncScope::ID SSID,
                                                BasicBlock *InsertAtEnd);

CINDEX_LINKAGE StoreInst *
StoreInst_CreateDefaultAtEnd(Value *Val, Value *Ptr,
                             bool isVolatile, BasicBlock *InsertAtEnd);
CINDEX_LINKAGE void StoreInstSetAlignment(StoreInst *inst, Align align);

// LoadInst : Instruction
CINDEX_LINKAGE LoadInst *
LoadInst_CreateBefore(Type *Ty, Value *Ptr, const char *NameStr,
                      bool isVolatile, Align Align, AtomicOrdering Order,
                      SyncScope::ID SSID = SyncScope::System,
                      Instruction *InsertBefore = nullptr);
CINDEX_LINKAGE LoadInst *
LoadInst_CreateDefaultBefore(Type *type, Value *arraySize, const char *name,
                               Instruction *insertBefore);
CINDEX_LINKAGE LoadInst *
LoadInst_CreateAtEnd(Type *Ty, Value *Ptr, const char *NameStr, bool isVolatile,
                     Align Align, AtomicOrdering Order, SyncScope::ID SSID,
                     BasicBlock *InsertAtEnd);
CINDEX_LINKAGE LoadInst *LoadInst_CreateDefaultAtEnd(Type *Ty, Value *Ptr,
                                                     const char *NameStr,
                                                     bool isVolatile,
                                                     BasicBlock *InsertAtEnd);
CINDEX_LINKAGE void LoadInstSetAlignment(LoadInst *inst, Align align);

// OperandBundleDef
CINDEX_LINKAGE void OperandBundleDef_Constuctor(Value **bundle,
                                                int bundleHiIndex, char *Tag,
                                                OperandBundleDef *out);

CINDEX_LINKAGE void OperandBundleDef_Destructor(OperandBundleDef *out);

CINDEX_LINKAGE int GetAlignSize(int t);

CINDEX_LINKAGE bool Generate(ILLVMModule *m, IStream * stream);
CINDEX_LINKAGE TargetOptions DefaultTargetOptions();

LLVM_CLANG_C_EXTERN_C_END

#endif