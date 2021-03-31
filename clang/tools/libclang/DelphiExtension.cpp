
#include "clang-c/DelphiExtension.h"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <string>
#include <llvm-c/Target.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/InitializePasses.h>
#include <llvm/CodeGen/CommandFlags.h>

class ImplUnknown : public IUnknown {
private:
  long m_cRef;

public:
  ImplUnknown() : m_cRef(0) {}
  virtual ~ImplUnknown() {}
  HRESULT __stdcall QueryInterface(const GUID &riid, void **ppvObject) override {
    return E_NOINTERFACE;
  }

  ULONG __stdcall AddRef(void) override {
    InterlockedIncrement(&m_cRef);
    return m_cRef;
  }

  ULONG __stdcall Release(void) override {
    // Decrement the object's internal counter.
    ULONG ulRefCount = InterlockedDecrement(&m_cRef);
    if (0 == m_cRef) {
      delete this;
    }
    return ulRefCount;
  }
};

class ImplLLVMModule;

class ImplLLVMContext : public LLVMContext,
                        public ILLVMContext,
                        public ImplUnknown {
public:
  virtual ~ImplLLVMContext() {}
  HRESULT __stdcall QueryInterface(const GUID &riid,
                                   void **ppvObject) override {
    return ImplUnknown::QueryInterface(riid, ppvObject);
  }
  ULONG __stdcall AddRef(void) override { return ImplUnknown::AddRef(); }
  ULONG __stdcall Release(void) override { return ImplUnknown::Release(); }

  ILLVMModule __stdcall *CreateModule(const char *moduleID) override;

  IntegerType __stdcall *GetIntegerType(unsigned int numBits) override {
    return IntegerType::get(*this, numBits);
  }

  FunctionType __stdcall *GetFunctionType(Type *result, Type **params,
                                          int paramsHiIndex,
                                          bool isVarArg) override {
    return FunctionType::get(
        result, ArrayRef<Type *>(params, paramsHiIndex + 1), isVarArg);
  }

  BasicBlock __stdcall *CreateBasicBlock(const char *name, Function *parent,
                                         BasicBlock *insertBefore) override {
    return BasicBlock::Create(*this, Twine(name), parent, insertBefore);
  }

  void __stdcall CreateBuilder(IIRBuilder **builder) override;
};

class ImplIRBuilder : public IRBuilder<>,
                      public IIRBuilder,
                      public ImplUnknown {
private:
  ImplLLVMContext *m_Context;

public:
  ImplIRBuilder(ImplLLVMContext *C, MDNode *FPMathTag = nullptr,
                ArrayRef<OperandBundleDef> OpBundles = None)
      : IRBuilder<>(*C, FPMathTag, OpBundles), m_Context(C) {
    m_Context->AddRef();
  }
  virtual ~ImplIRBuilder() { m_Context->Release(); }

  HRESULT __stdcall QueryInterface(const GUID &riid,
                                   void **ppvObject) override {
    return ImplUnknown::QueryInterface(riid, ppvObject);
  }
  ULONG __stdcall AddRef(void) override { return ImplUnknown::AddRef(); }
  ULONG __stdcall Release(void) override { return ImplUnknown::Release(); }

  /// Insert and return the specified instruction.
  Instruction __stdcall *Insert(Instruction *I, const char *Name) override {
    return IRBuilder<>::Insert(I, Twine(Name ? Name : ""));
  }

  /// No-op overload to handle constants.
  Constant __stdcall *Insert(Constant *C, const char *Name) override {
    return IRBuilder<>::Insert(C, Twine(Name ? Name : ""));
  }

  Value __stdcall *Insert(Value *V, const char *Name) override {
    return IRBuilder<>::Insert(V, Twine(Name ? Name : ""));
  }

  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//

  /// Clear the insertion point: created instructions will not be
  /// inserted into a block.
  void __stdcall ClearInsertionPoint() override {
    IRBuilder<>::ClearInsertionPoint();
  }

  BasicBlock __stdcall *GetInsertBlock() override {
    return IRBuilder<>::GetInsertBlock();
  }
  // BasicBlock::iterator GetInsertPoint() const { return InsertPt; }
  void __stdcall GetContext(ILLVMContext **context) override {
    *context = m_Context;
    m_Context->AddRef();
  }

  /// This specifies that created instructions should be appended to the
  /// end of the specified block.
  void __stdcall SetInsertPoint(BasicBlock *TheBB) override {
    IRBuilder<>::SetInsertPoint(TheBB);
  }

  /// This specifies that created instructions should be inserted before
  /// the specified instruction.
  void __stdcall SetInsertPoint(Instruction *I) override {
    IRBuilder<>::SetInsertPoint(I);
  }

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  /*void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
    if (IP != TheBB->end())
      SetCurrentDebugLocation(IP->getDebugLoc());
  }*/

  /// Set location information used by debugging information.
  void __stdcall SetCurrentDebugLocation(DebugLoc L) override {
    IRBuilder<>::SetCurrentDebugLocation(L);
  }

  /// Collect metadata with IDs \p MetadataKinds from \p Src which should be
  /// added to all created instructions. Entries present in MedataDataToCopy but
  /// not on \p Src will be dropped from MetadataToCopy.
  /*void CollectMetadataToCopy(Instruction *Src,
                             ArrayRef<unsigned> MetadataKinds) {
    for (unsigned K : MetadataKinds)
      AddOrRemoveMetadataToCopy(K, Src->getMetadata(K));
  }*/

  /// Get location information used by debugging information.
  DebugLoc __stdcall getCurrentDebugLocation() override {
    return IRBuilder<>::getCurrentDebugLocation();
  }

  /// If this builder has a current debug location, set it on the
  /// specified instruction.
  void __stdcall SetInstDebugLocation(Instruction *I) const override {
    IRBuilder<>::SetInstDebugLocation(I);
  }

  /// Add all entries in MetadataToCopy to \p I.
  void __stdcall AddMetadataToInst(Instruction *I) const override {
    IRBuilder<>::AddMetadataToInst(I);
  }

  /// Get the return type of the current function that we're emitting
  /// into.
  Type __stdcall *getCurrentFunctionReturnType() override {
    return IRBuilder<>::getCurrentFunctionReturnType();
  }

  /// Returns the current insert point.
  InsertPoint __stdcall saveIP() override { return IRBuilder<>::saveIP(); }

  /// Returns the current insert point, clearing it in the process.
  InsertPoint __stdcall saveAndClearIP() override {
    return IRBuilder<>::saveAndClearIP();
  }

  /// Sets the current insert point to a previously-saved location.
  void __stdcall restoreIP(InsertPoint IP) override {
    IRBuilder<>::restoreIP(IP);
  }

  /// Get the floating point math metadata being used.
  // MDNode *getDefaultFPMathTag() const { return DefaultFPMathTag; }

  /// Get the flags to be applied to created floating point ops
  // FastMathFlags getFastMathFlags() const { return FMF; }

  // FastMathFlags &getFastMathFlags() { return FMF; }

  /// Clear the fast-math flags.
  // void clearFastMathFlags() { FMF.clear(); }

  /// Set the floating point math metadata to be used.
  // void setDefaultFPMathTag(MDNode *FPMathTag) { DefaultFPMathTag = FPMathTag;
  // }

  /// Set the fast-math flags to be used with generated fp-math operators
  // void setFastMathFlags(FastMathFlags NewFMF) { FMF = NewFMF; }

  /// Enable/Disable use of constrained floating point math. When
  /// enabled the CreateF<op>() calls instead create constrained
  /// floating point intrinsic calls. Fast math flags are unaffected
  /// by this setting.
  // void setIsFPConstrained(bool IsCon) { IsFPConstrained = IsCon; }

  /// Query for the use of constrained floating point math
  // bool getIsFPConstrained() { return IsFPConstrained; }

  /// Set the exception handling to be used with constrained floating point
  /*void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    Optional<StringRef> ExceptStr = ExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr.hasValue() && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }*/

  /// Set the rounding mode handling to be used with constrained floating point
  /*void setDefaultConstrainedRounding(RoundingMode NewRounding) {
#ifndef NDEBUG
    Optional<StringRef> RoundingStr = RoundingModeToStr(NewRounding);
    assert(RoundingStr.hasValue() && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }*/

  /// Get the exception handling used with constrained floating point
  /*fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }*/

  /// Get the rounding mode handling used with constrained floating point
  /*RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }*/

  /*void setConstrainedFPFunctionAttr() {
    assert(BB && "Must have a basic block to set any function attributes!");

    Function *F = BB->getParent();
    if (!F->hasFnAttribute(Attribute::StrictFP)) {
      F->addFnAttr(Attribute::StrictFP);
    }
  }*/

  /*void setConstrainedFPCallAttr(CallBase *I) {
    I->addAttribute(AttributeList::FunctionIndex, Attribute::StrictFP);
  }*/

  /*void setDefaultOperandBundles(ArrayRef<OperandBundleDef> OpBundles) {
    DefaultOperandBundles = OpBundles;
  }*/

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
  ConstantInt __stdcall *getInt1(bool V) override {
    return IRBuilder<>::getInt1(V);
  }

  /// Get the constant value for i1 true.
  ConstantInt __stdcall *getTrue() override { return IRBuilder<>::getTrue(); }

  /// Get the constant value for i1 false.
  ConstantInt __stdcall *getFalse() override { return IRBuilder<>::getFalse(); }

  /// Get a constant 8-bit value.
  ConstantInt __stdcall *getInt8(uint8_t C) override {
    return IRBuilder<>::getInt8(C);
  }

  /// Get a constant 16-bit value.
  ConstantInt __stdcall *getInt16(uint16_t C) override {
    return IRBuilder<>::getInt16(C);
  }

  /// Get a constant 32-bit value.
  ConstantInt __stdcall *getInt32(uint32_t C) override {
    return IRBuilder<>::getInt32(C);
  }

  /// Get a constant 64-bit value.
  ConstantInt __stdcall *getInt64(uint64_t C) override {
    return IRBuilder<>::getInt64(C);
  }

  /// Get a constant N-bit value, zero extended or truncated from
  /// a 64-bit value.
  ConstantInt __stdcall *getIntN(unsigned N, uint64_t C) override {
    return IRBuilder<>::getIntN(N, C);
  }

  /// Get a constant integer value.
  /*ConstantInt *getInt(const APInt &AI) {
    return ConstantInt::get(Context, AI);
  }*/

  //===--------------------------------------------------------------------===//
  // Type creation methods
  //===--------------------------------------------------------------------===//

  /// Fetch the type representing a single bit
  IntegerType __stdcall *getInt1Ty() override {
    return IRBuilder<>::getInt1Ty();
  }

  /// Fetch the type representing an 8-bit integer.
  IntegerType __stdcall *getInt8Ty() override {
    return IRBuilder<>::getInt8Ty();
  }

  /// Fetch the type representing a 16-bit integer.
  IntegerType __stdcall *getInt16Ty() override {
    return IRBuilder<>::getInt16Ty();
  }

  /// Fetch the type representing a 32-bit integer.
  IntegerType __stdcall *getInt32Ty() override {
    return IRBuilder<>::getInt32Ty();
  }

  /// Fetch the type representing a 64-bit integer.
  IntegerType __stdcall *getInt64Ty() override {
    return IRBuilder<>::getInt64Ty();
  }

  /// Fetch the type representing a 128-bit integer.
  IntegerType __stdcall *getInt128Ty() override {
    return IRBuilder<>::getInt128Ty();
  }

  /// Fetch the type representing an N-bit integer.
  IntegerType __stdcall *getIntNTy(unsigned N) override {
    return IRBuilder<>::getIntNTy(N);
  }

  /// Fetch the type representing a 16-bit floating point value.
  Type __stdcall *getHalfTy() override { return IRBuilder<>::getHalfTy(); }

  /// Fetch the type representing a 16-bit brain floating point value.
  Type __stdcall *getBFloatTy() override { return IRBuilder<>::getBFloatTy(); }

  /// Fetch the type representing a 32-bit floating point value.
  Type __stdcall *getFloatTy() override { return IRBuilder<>::getFloatTy(); }

  /// Fetch the type representing a 64-bit floating point value.
  Type __stdcall *getDoubleTy() override { return IRBuilder<>::getDoubleTy(); }

  /// Fetch the type representing void.
  Type __stdcall *getVoidTy() override { return IRBuilder<>::getVoidTy(); }

  /// Fetch the type representing a pointer to an 8-bit integer value.
  PointerType __stdcall *getInt8PtrTy(unsigned AddrSpace = 0) override {
    return IRBuilder<>::getInt8PtrTy(AddrSpace);
  }

  /// Fetch the type representing a pointer to an integer value.
  IntegerType __stdcall *getIntPtrTy(const DataLayout &DL,
                                     unsigned AddrSpace = 0) override {
    return IRBuilder<>::getIntPtrTy(DL, AddrSpace);
  }

  //===--------------------------------------------------------------------===//
  // Intrinsic creation methods
  //===--------------------------------------------------------------------===//

  /// Create and insert a memset to the specified pointer and the
  /// specified value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst __stdcall *CreateMemSet(Value *Ptr, Value *Val, uint64_t Size,
                                   MaybeAlign Align, bool isVolatile = false,
                                   MDNode *TBAATag = nullptr,
                                   MDNode *ScopeTag = nullptr,
                                   MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemSet(Ptr, Val, Size, Align, isVolatile, TBAATag,
                                     ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *CreateMemSet(Value *Ptr, Value *Val, Value *Size,
                                   MaybeAlign Align, bool isVolatile = false,
                                   MDNode *TBAATag = nullptr,
                                   MDNode *ScopeTag = nullptr,
                                   MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemSet(Ptr, Val, Size, Align, isVolatile, TBAATag,
                                     ScopeTag, NoAliasTag);
  }

  /// Create and insert an element unordered-atomic memset of the region of
  /// memory starting at the given pointer to the given value.
  ///
  /// If the pointer isn't an i8*, it will be converted. If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst __stdcall *CreateElementUnorderedAtomicMemSet(
      Value *Ptr, Value *Val, uint64_t Size, Align Alignment,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateElementUnorderedAtomicMemSet(
        Ptr, Val, Size, Align(Alignment), ElementSize, TBAATag, ScopeTag,
        NoAliasTag);
  }

  CallInst __stdcall *CreateElementUnorderedAtomicMemSet(
      Value *Ptr, Value *Val, Value *Size, Align Alignment,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateElementUnorderedAtomicMemSet(
        Ptr, Val, Size, Align(Alignment), ElementSize, TBAATag, ScopeTag,
        NoAliasTag);
  }

  /// Create and insert a memcpy between the specified pointers.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst __stdcall *CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src,
                                   MaybeAlign SrcAlign, uint64_t Size,
                                   bool isVolatile = false,
                                   MDNode *TBAATag = nullptr,
                                   MDNode *TBAAStructTag = nullptr,
                                   MDNode *ScopeTag = nullptr,
                                   MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemCpy(Dst, DstAlign, Src, SrcAlign, Size,
                                     isVolatile, TBAATag, TBAAStructTag,
                                     ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *CreateMemTransferInst(
      Intrinsic::ID IntrID, Value *Dst, MaybeAlign DstAlign, Value *Src,
      MaybeAlign SrcAlign, Value *Size, bool isVolatile = false,
      MDNode *TBAATag = nullptr, MDNode *TBAAStructTag = nullptr,
      MDNode *ScopeTag = nullptr, MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemTransferInst(
        IntrID, Dst, DstAlign, Src, SrcAlign, Size, isVolatile, TBAATag,
        TBAAStructTag, ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *
  CreateMemCpy(Value *Dst, MaybeAlign DstAlign, Value *Src, MaybeAlign SrcAlign,
               Value *Size, bool isVolatile = false, MDNode *TBAATag = nullptr,
               MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
               MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemCpy(Dst, DstAlign, Src, SrcAlign, Size,
                                     isVolatile, TBAATag, TBAAStructTag,
                                     ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *CreateMemCpyInline(Value *Dst, MaybeAlign DstAlign,
                                         Value *Src, MaybeAlign SrcAlign,
                                         Value *Size) override {
    return IRBuilder<>::CreateMemCpyInline(Dst, DstAlign, Src, SrcAlign, Size);
  }

  /// Create and insert an element unordered-atomic memcpy between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers,
  /// respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst __stdcall *CreateElementUnorderedAtomicMemCpy(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateElementUnorderedAtomicMemCpy(
        Dst, DstAlign, Src, SrcAlign, Size, ElementSize, TBAATag, TBAAStructTag,
        ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                                    MaybeAlign SrcAlign, uint64_t Size,
                                    bool isVolatile = false,
                                    MDNode *TBAATag = nullptr,
                                    MDNode *ScopeTag = nullptr,
                                    MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemMove(Dst, DstAlign, Src, SrcAlign, Size,
                                      isVolatile, TBAATag, ScopeTag,
                                      NoAliasTag);
  }

  CallInst __stdcall *CreateMemMove(Value *Dst, MaybeAlign DstAlign, Value *Src,
                                    MaybeAlign SrcAlign, Value *Size,
                                    bool isVolatile = false,
                                    MDNode *TBAATag = nullptr,
                                    MDNode *ScopeTag = nullptr,
                                    MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateMemMove(Dst, DstAlign, Src, SrcAlign, Size,
                                      isVolatile, TBAATag, ScopeTag,
                                      NoAliasTag);
  }

  /// \brief Create and insert an element unordered-atomic memmove between the
  /// specified pointers.
  ///
  /// DstAlign/SrcAlign are the alignments of the Dst/Src pointers,
  /// respectively.
  ///
  /// If the pointers aren't i8*, they will be converted.  If a TBAA tag is
  /// specified, it will be added to the instruction. Likewise with alias.scope
  /// and noalias tags.
  CallInst __stdcall *CreateElementUnorderedAtomicMemMove(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, uint64_t Size,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateElementUnorderedAtomicMemMove(
        Dst, DstAlign, Src, SrcAlign, IRBuilder<>::getInt64(Size), ElementSize,
        TBAATag, TBAAStructTag, ScopeTag, NoAliasTag);
  }

  CallInst __stdcall *CreateElementUnorderedAtomicMemMove(
      Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
      uint32_t ElementSize, MDNode *TBAATag = nullptr,
      MDNode *TBAAStructTag = nullptr, MDNode *ScopeTag = nullptr,
      MDNode *NoAliasTag = nullptr) override {
    return IRBuilder<>::CreateElementUnorderedAtomicMemMove(
        Dst, DstAlign, Src, SrcAlign, Size, ElementSize, TBAATag, TBAAStructTag,
        ScopeTag, NoAliasTag);
  }

  /// Create a vector fadd reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value for ordered reductions.
  CallInst __stdcall *CreateFAddReduce(Value *Acc, Value *Src) override {
    return IRBuilder<>::CreateFAddReduce(Acc, Src);
  }

  /// Create a vector fmul reduction intrinsic of the source vector.
  /// The first parameter is a scalar accumulator value for ordered reductions.
  CallInst __stdcall *CreateFMulReduce(Value *Acc, Value *Src) override {
    return IRBuilder<>::CreateFMulReduce(Acc, Src);
  }

  /// Create a vector int add reduction intrinsic of the source vector.
  CallInst __stdcall *CreateAddReduce(Value *Src) override {
    return IRBuilder<>::CreateAddReduce(Src);
  }

  /// Create a vector int mul reduction intrinsic of the source vector.
  CallInst __stdcall *CreateMulReduce(Value *Src) override {
    return IRBuilder<>::CreateMulReduce(Src);
  }

  /// Create a vector int AND reduction intrinsic of the source vector.
  CallInst __stdcall *CreateAndReduce(Value *Src) override {
    return IRBuilder<>::CreateAndReduce(Src);
  }

  /// Create a vector int OR reduction intrinsic of the source vector.
  CallInst __stdcall *CreateOrReduce(Value *Src) override {
    return IRBuilder<>::CreateOrReduce(Src);
  }

  /// Create a vector int XOR reduction intrinsic of the source vector.
  CallInst __stdcall *CreateXorReduce(Value *Src) override {
    return IRBuilder<>::CreateXorReduce(Src);
  }

  /// Create a vector integer max reduction intrinsic of the source
  /// vector.
  CallInst __stdcall *CreateIntMaxReduce(Value *Src,
                                         bool IsSigned = false) override {
    return IRBuilder<>::CreateIntMaxReduce(Src, IsSigned);
  }

  /// Create a vector integer min reduction intrinsic of the source
  /// vector.
  CallInst __stdcall *CreateIntMinReduce(Value *Src,
                                         bool IsSigned = false) override {
    return IRBuilder<>::CreateIntMinReduce(Src, IsSigned);
  }

  /// Create a vector float max reduction intrinsic of the source
  /// vector.
  CallInst __stdcall *CreateFPMaxReduce(Value *Src) override {
    return IRBuilder<>::CreateFPMaxReduce(Src);
  }

  /// Create a vector float min reduction intrinsic of the source
  /// vector.
  CallInst __stdcall *CreateFPMinReduce(Value *Src) override {
    return IRBuilder<>::CreateFPMinReduce(Src);
  }

  /// Create a lifetime.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst __stdcall *
  CreateLifetimeStart(Value *Ptr, ConstantInt *Size = nullptr) override {
    return IRBuilder<>::CreateLifetimeStart(Ptr, Size);
  }

  /// Create a lifetime.end intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst __stdcall *CreateLifetimeEnd(Value *Ptr,
                                        ConstantInt *Size = nullptr) override {
    return IRBuilder<>::CreateLifetimeEnd(Ptr, Size);
  }

  /// Create a call to invariant.start intrinsic.
  ///
  /// If the pointer isn't i8* it will be converted.
  CallInst __stdcall *
  CreateInvariantStart(Value *Ptr, ConstantInt *Size = nullptr) override {
    return IRBuilder<>::CreateInvariantStart(Ptr, Size);
  }

  /// Create a call to Masked Load intrinsic
  CallInst __stdcall *CreateMaskedLoad(Value *Ptr, Align Alignment, Value *Mask,
                                       Value *PassThru,
                                       const char *Name) override {
    return IRBuilder<>::CreateMaskedLoad(Ptr, Alignment, Mask, PassThru,
                                         Name ? Name : "");
  }

  /// Create a call to Masked Store intrinsic
  CallInst __stdcall *CreateMaskedStore(Value *Val, Value *Ptr, Align Alignment,
                                        Value *Mask) override {
    return IRBuilder<>::CreateMaskedStore(Val, Ptr, Alignment, Mask);
  }

  /// Create a call to Masked Gather intrinsic
  CallInst __stdcall *CreateMaskedGather(Value *Ptrs, Align Alignment,
                                         Value *Mask, Value *PassThru,
                                         const char *Name) override {
    return IRBuilder<>::CreateMaskedGather(Ptrs, Alignment, Mask, PassThru,
                                           Twine(Name ? Name : ""));
  }

  /// Create a call to Masked Scatter intrinsic
  CallInst __stdcall *CreateMaskedScatter(Value *Val, Value *Ptrs,
                                          Align Alignment,
                                          Value *Mask = nullptr) override {
    return IRBuilder<>::CreateMaskedScatter(Val, Ptrs, Alignment, Mask);
  }

  /// Create an assume intrinsic call that allows the optimizer to
  /// assume that the provided condition will be true.
  ///
  /// The optional argument \p OpBundles specifies operand bundles that are
  /// added to the call instruction.
  /*CallInst __stdcall *
  CreateAssumption(Value *Cond,
                   ArrayRef<OperandBundleDef> OpBundles = None) override {
    return IRBuilder<>::CreateAssumption(Cond, OpBundles);
  }*/

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  CallInst __stdcall *
  CreateGCStatepointCall(uint64_t ID, uint32_t NumPatchBytes,
                         Value *ActualCallee, Value **CallArgs,
                         int CallArgsHiIndex, Value **DeoptArgs,
                         int DeoptArgsHiIndex, Value **GCArgs,
                         int GCArgsHiIndex, const char *Name) override {
    return IRBuilder<>::CreateGCStatepointCall(
        ID, NumPatchBytes, ActualCallee,
        ArrayRef<Value *>(CallArgs, CallArgsHiIndex + 1),
        ArrayRef<Value *>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Create a call to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  CallInst __stdcall *CreateGCStatepointCall(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualCallee, uint32_t Flags,
      Value **CallArgs, int CallArgsHiIndex, Use *TransitionArgs,
      int TransitionArgsHiIndex, Use *DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name) override {
    return IRBuilder<>::CreateGCStatepointCall(
        ID, NumPatchBytes, ActualCallee, Flags,
        ArrayRef<Value *>(CallArgs, CallArgsHiIndex + 1),
        ArrayRef<Use>(TransitionArgs, TransitionArgsHiIndex + 1),
        ArrayRef<Use>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Conveninence function for the common case when CallArgs are filled
  /// in using makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be
  /// .get()'ed to get the Value pointer.
  CallInst __stdcall *CreateGCStatepointCall(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualCallee, Use *CallArgs,
      int CallArgsHiIndex, Value **DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateGCStatepointCall(
        ID, NumPatchBytes, ActualCallee,
        ArrayRef<Use>(CallArgs, CallArgsHiIndex + 1),
        DeoptArgsHiIndex == -1
            ? None
            : ArrayRef<Value *>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, Value **InvokeArgs,
      int InvokeArgsHiIndex, Value **DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateGCStatepointInvoke(
        ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest,
        ArrayRef<Value *>(InvokeArgs, InvokeArgsHiIndex + 1),
        DeoptArgsHiIndex == -1
            ? None
            : ArrayRef<Value *>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Create an invoke to the experimental.gc.statepoint intrinsic to
  /// start a new statepoint sequence.
  InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, uint32_t Flags,
      Value **InvokeArgs, int InvokeArgsHiIndex, Use *TransitionArgs,
      int TransitionArgsHiIndex, Use *DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateGCStatepointInvoke(
        ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest, Flags,
        ArrayRef<Value *>(InvokeArgs, InvokeArgsHiIndex + 1),
        TransitionArgsHiIndex == -1
            ? None
            : ArrayRef<Use>(TransitionArgs, TransitionArgsHiIndex + 1),
        DeoptArgsHiIndex == -1 ? None
                               : ArrayRef<Use>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  // Convenience function for the common case when CallArgs are filled in using
  // makeArrayRef(CS.arg_begin(), CS.arg_end()); Use needs to be .get()'ed to
  // get the Value *.
  InvokeInst __stdcall *CreateGCStatepointInvoke(
      uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
      BasicBlock *NormalDest, BasicBlock *UnwindDest, Use *InvokeArgs,
      int InvokeArgsHiIndex, Value **DeoptArgs, int DeoptArgsHiIndex,
      Value **GCArgs, int GCArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateGCStatepointInvoke(
        ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest,
        ArrayRef<Use>(InvokeArgs, InvokeArgsHiIndex + 1),
        DeoptArgsHiIndex == -1
            ? None
            : ArrayRef<Value *>(DeoptArgs, DeoptArgsHiIndex + 1),
        ArrayRef<Value *>(GCArgs, GCArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Create a call to the experimental.gc.result intrinsic to extract
  /// the result from a call wrapped in a statepoint.
  CallInst __stdcall *CreateGCResult(Instruction *Statepoint, Type *ResultType,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateGCResult(Statepoint, ResultType,
                                       Twine(Name ? Name : ""));
  }

  /// Create a call to the experimental.gc.relocate intrinsics to
  /// project the relocated value of one pointer from the statepoint.
  CallInst __stdcall *CreateGCRelocate(Instruction *Statepoint, int BaseOffset,
                                       int DerivedOffset, Type *ResultType,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateGCRelocate(Statepoint, BaseOffset, DerivedOffset,
                                         ResultType, Twine(Name ? Name : ""));
  }

  /// Create a call to llvm.vscale, multiplied by \p Scaling. The type of VScale
  /// will be the same type as that of \p Scaling.
  Value __stdcall *CreateVScale(Constant *Scaling,
                                const char *Name = "") override {
    return IRBuilder<>::CreateVScale(Scaling, Twine(Name ? Name : ""));
  }

  /// Create a call to intrinsic \p ID with 1 operand which is mangled on its
  /// type.
  CallInst __stdcall *CreateUnaryIntrinsic(Intrinsic::ID ID, Value *V,
                                           Instruction *FMFSource = nullptr,
                                           const char *Name = "") override {
    return IRBuilder<>::CreateUnaryIntrinsic(ID, V, FMFSource,
                                             Twine(Name ? Name : ""));
  }

  /// Create a call to intrinsic \p ID with 2 operands which is mangled on the
  /// first type.
  CallInst __stdcall *CreateBinaryIntrinsic(Intrinsic::ID ID, Value *LHS,
                                            Value *RHS,
                                            Instruction *FMFSource = nullptr,
                                            const char *Name = "") override {
    return IRBuilder<>::CreateBinaryIntrinsic(ID, LHS, RHS, FMFSource,
                                              Twine(Name ? Name : ""));
  }

  /// Create a call to intrinsic \p ID with \p args, mangled using \p Types. If
  /// \p FMFSource is provided, copy fast-math-flags from that instruction to
  /// the intrinsic.
  CallInst __stdcall *CreateIntrinsic(Intrinsic::ID ID, Type **Types,
                                      int TypesHiIndex, Value **Args,
                                      int ArgsHiIndex,
                                      Instruction *FMFSource = nullptr,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateIntrinsic(
        ID, ArrayRef<Type *>(Types, TypesHiIndex + 1),
        ArrayRef<Value *>(Args, ArgsHiIndex + 1), FMFSource,
        Twine(Name ? Name : ""));
  }

  /// Create call to the minnum intrinsic.
  CallInst __stdcall *CreateMinNum(Value *LHS, Value *RHS,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateMinNum(LHS, RHS, Twine(Name ? Name : ""));
  }

  /// Create call to the maxnum intrinsic.
  CallInst __stdcall *CreateMaxNum(Value *LHS, Value *RHS,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateMaxNum(LHS, RHS, Twine(Name ? Name : ""));
  }

  /// Create call to the minimum intrinsic.
  CallInst __stdcall *CreateMinimum(Value *LHS, Value *RHS,
                                    const char *Name = "") override {
    return IRBuilder<>::CreateMinimum(LHS, RHS, Twine(Name ? Name : ""));
  }

  /// Create call to the maximum intrinsic.
  CallInst __stdcall *CreateMaximum(Value *LHS, Value *RHS,
                                    const char *Name = "") override {
    return IRBuilder<>::CreateMaximum(LHS, RHS, Twine(Name ? Name : ""));
  }

  /// Create a call to the experimental.vector.extract intrinsic.
  CallInst __stdcall *CreateExtractVector(Type *DstType, Value *SrcVec,
                                          Value *Idx,
                                          const char *Name = "") override {
    return IRBuilder<>::CreateExtractVector(DstType, SrcVec, Idx,
                                            Twine(Name ? Name : ""));
  }

  /// Create a call to the experimental.vector.insert intrinsic.
  CallInst __stdcall *CreateInsertVector(Type *DstType, Value *SrcVec,
                                         Value *SubVec, Value *Idx,
                                         const char *Name = "") override {
    return IRBuilder<>::CreateInsertVector(DstType, SrcVec, SubVec, Idx,
                                           Twine(Name ? Name : ""));
  }

  /// Create a 'ret void' instruction.
  ReturnInst __stdcall *CreateRetVoid() override {
    return IRBuilder<>::CreateRetVoid();
  }

  /// Create a 'ret <val>' instruction.
  ReturnInst __stdcall *CreateRet(Value *V) override {
    return IRBuilder<>::CreateRet(V);
  }

  /// Create a sequence of N insertvalue instructions,
  /// with one Value from the retVals array each, that build a aggregate
  /// return value one value at a time, and a ret instruction to return
  /// the resulting aggregate value.
  ///
  /// This is a convenience function for code that uses aggregate return values
  /// as a vehicle for having multiple return values.
  ReturnInst __stdcall *CreateAggregateRet(Value *const *retVals,
                                           int retValsHiIndex) override {
    return IRBuilder<>::CreateAggregateRet(retVals, retValsHiIndex + 1);
  }

  /// Create an unconditional 'br label X' instruction.
  BranchInst __stdcall *CreateBr(BasicBlock *Dest) override {
    return IRBuilder<>::CreateBr(Dest);
  }

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst __stdcall *CreateCondBr(Value *Cond, BasicBlock *True,
                                     BasicBlock *False,
                                     MDNode *BranchWeights = nullptr,
                                     MDNode *Unpredictable = nullptr) override {
    return IRBuilder<>::CreateCondBr(Cond, True, False, BranchWeights,
                                     Unpredictable);
  }

  /// Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction. Copy branch meta data if available.
  BranchInst __stdcall *CreateCondBr(Value *Cond, BasicBlock *True,
                                     BasicBlock *False,
                                     Instruction *MDSrc) override {
    return IRBuilder<>::CreateCondBr(Cond, True, False, MDSrc);
  }

  /// Create a switch instruction with the specified value, default dest,
  /// and with a hint for the number of cases that will be added (for efficient
  /// allocation).
  SwitchInst __stdcall *CreateSwitch(Value *V, BasicBlock *Dest,
                                     unsigned NumCases = 10,
                                     MDNode *BranchWeights = nullptr,
                                     MDNode *Unpredictable = nullptr) override {
    return IRBuilder<>::CreateSwitch(V, Dest, NumCases, BranchWeights,
                                     Unpredictable);
  }

  /// Create an indirect branch instruction with the specified address
  /// operand, with an optional hint for the number of destinations that will be
  /// added (for efficient allocation).
  IndirectBrInst __stdcall *CreateIndirectBr(Value *Addr,
                                             unsigned NumDests = 10) override {
    return IRBuilder<>::CreateIndirectBr(Addr, NumDests);
  }

  /// Create an invoke instruction.
  InvokeInst __stdcall *
  CreateInvoke(FunctionType *Ty, Value *Callee, BasicBlock *NormalDest,
               BasicBlock *UnwindDest, Value **Args, int ArgsHiIndex,
               OperandBundleDef *OpBundles, int OpBundlesHiIndex,
               const char *Name = "") override {
    return IRBuilder<>::CreateInvoke(
        Ty, Callee, NormalDest, UnwindDest,
        ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  InvokeInst __stdcall *CreateInvoke(FunctionType *Ty, Value *Callee,
                                     BasicBlock *NormalDest,
                                     BasicBlock *UnwindDest, Value **Args,
                                     int ArgsHiIndex,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInvoke(Ty, Callee, NormalDest, UnwindDest,
                                     ArrayRef<Value *>(Args, ArgsHiIndex + 1),
                                     Twine(Name ? Name : ""));
  }

  InvokeInst __stdcall *
  CreateInvoke(FunctionCallee Callee, BasicBlock *NormalDest,
               BasicBlock *UnwindDest, Value **Args, int ArgsHiIndex,
               OperandBundleDef *OpBundles, int OpBundlesHiIndex,
               const char *Name = "") override {
    return IRBuilder<>::CreateInvoke(
        Callee, NormalDest, UnwindDest,
        ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  InvokeInst __stdcall *CreateInvoke(FunctionCallee Callee,
                                     BasicBlock *NormalDest,
                                     BasicBlock *UnwindDest, Value **Args,
                                     int ArgsHiIndex,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInvoke(Callee, NormalDest, UnwindDest,
                                     ArrayRef<Value *>(Args, ArgsHiIndex + 1),
                                     Twine(Name ? Name : ""));
  }

  /// \brief Create a callbr instruction.
  CallBrInst __stdcall *
  CreateCallBr(FunctionType *Ty, Value *Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateCallBr(
        Ty, Callee, DefaultDest,
        ArrayRef<BasicBlock *>(IndirectDests, IndirectDestsHiIndex + 1),
        ArrayRef<Value *>(Args, ArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  CallBrInst __stdcall *
  CreateCallBr(FunctionType *Ty, Value *Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, OperandBundleDef *OpBundles,
               int OpBundlesHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateCallBr(
        Ty, Callee, DefaultDest,
        ArrayRef<BasicBlock *>(IndirectDests, IndirectDestsHiIndex + 1),
        ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  CallBrInst __stdcall *
  CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateCallBr(
        Callee, DefaultDest,
        ArrayRef<BasicBlock *>(IndirectDests, IndirectDestsHiIndex + 1),
        ArrayRef<Value *>(Args, ArgsHiIndex + 1), Twine(Name ? Name : ""));
  }

  CallBrInst __stdcall *
  CreateCallBr(FunctionCallee Callee, BasicBlock *DefaultDest,
               BasicBlock **IndirectDests, int IndirectDestsHiIndex,
               Value **Args, int ArgsHiIndex, OperandBundleDef *OpBundles,
               int OpBundlesHiIndex, const char *Name = "") override {
    return IRBuilder<>::CreateCallBr(
        Callee, DefaultDest,
        ArrayRef<BasicBlock *>(IndirectDests, IndirectDestsHiIndex + 1),
        ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  ResumeInst __stdcall *CreateResume(Value *Exn) override {
    return IRBuilder<>::CreateResume(Exn);
  }

  CleanupReturnInst __stdcall *
  CreateCleanupRet(CleanupPadInst *CleanupPad,
                   BasicBlock *UnwindBB = nullptr) override {
    return IRBuilder<>::CreateCleanupRet(CleanupPad, UnwindBB);
  }

  CatchSwitchInst __stdcall *CreateCatchSwitch(Value *ParentPad,
                                               BasicBlock *UnwindBB,
                                               unsigned NumHandlers,
                                               const char *Name = "") override {
    return IRBuilder<>::CreateCatchSwitch(ParentPad, UnwindBB, NumHandlers,
                                          Twine(Name ? Name : ""));
  }

  CatchPadInst __stdcall *CreateCatchPad(Value *ParentPad, Value **Args,
                                         int ArgsHiIndex,
                                         const char *Name = "") override {
    return IRBuilder<>::CreateCatchPad(ParentPad,
                                       ArrayRef<Value *>(Args, ArgsHiIndex + 1),
                                       Twine(Name ? Name : ""));
  }

  CleanupPadInst __stdcall *CreateCleanupPad(Value *ParentPad, Value **Args,
                                             int ArgsHiIndex,
                                             const char *Name = "") override {
    return IRBuilder<>::CreateCleanupPad(
        ParentPad, ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  CatchReturnInst __stdcall *CreateCatchRet(CatchPadInst *CatchPad,
                                            BasicBlock *BB) override {
    return IRBuilder<>::CreateCatchRet(CatchPad, BB);
  }

  UnreachableInst __stdcall *CreateUnreachable() override {
    return IRBuilder<>::CreateUnreachable();
  }

  Value __stdcall *CreateAdd(Value *LHS, Value *RHS, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateAdd(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }

  Value __stdcall *CreateNSWAdd(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNSWAdd(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateNUWAdd(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNUWAdd(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSub(Value *LHS, Value *RHS, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateSub(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }

  Value __stdcall *CreateNSWSub(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNSWSub(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateNUWSub(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNUWSub(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateMul(Value *LHS, Value *RHS, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateMul(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }

  Value __stdcall *CreateNSWMul(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNSWMul(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateNUWMul(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateNUWMul(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateUDiv(Value *LHS, Value *RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateUDiv(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  Value __stdcall *CreateExactUDiv(Value *LHS, Value *RHS,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateExactUDiv(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSDiv(Value *LHS, Value *RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateSDiv(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  Value __stdcall *CreateExactSDiv(Value *LHS, Value *RHS,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateExactSDiv(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateURem(Value *LHS, Value *RHS,
                              const char *Name = "") override {
    return IRBuilder<>::CreateURem(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSRem(Value *LHS, Value *RHS,
                              const char *Name = "") override {
    return IRBuilder<>::CreateSRem(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateShl(Value *LHS, Value *RHS, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateShl(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }

  /*Value __stdcall *CreateShl(Value *LHS, const APInt &RHS,
                             const char *Name = "", bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateShl(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }*/

  Value __stdcall *CreateShl(Value *LHS, uint64_t RHS, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateShl(LHS, RHS, Twine(Name ? Name : ""), HasNUW,
                                  HasNSW);
  }

  Value __stdcall *CreateLShr(Value *LHS, Value *RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateLShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  /*Value __stdcall *CreateLShr(Value *LHS, const APInt &RHS,
                              const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateLShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }*/

  Value __stdcall *CreateLShr(Value *LHS, uint64_t RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateLShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  Value __stdcall *CreateAShr(Value *LHS, Value *RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateAShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  /*Value __stdcall *CreateAShr(Value *LHS, const APInt &RHS,
                              const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateAShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }*/

  Value __stdcall *CreateAShr(Value *LHS, uint64_t RHS, const char *Name = "",
                              bool isExact = false) override {
    return IRBuilder<>::CreateAShr(LHS, RHS, Twine(Name ? Name : ""), isExact);
  }

  Value __stdcall *CreateAnd(Value *LHS, Value *RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateAnd(LHS, RHS, Twine(Name ? Name : ""));
  }

  /*Value __stdcall *CreateAnd(Value *LHS, const APInt &RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateAnd(LHS, RHS, Twine(Name ? Name : ""));
  }*/

  Value __stdcall *CreateAnd(Value *LHS, uint64_t RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateAnd(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateAnd(Value **Ops, int OpsHiIndex) override {
    return IRBuilder<>::CreateAnd(ArrayRef<Value *>(Ops, OpsHiIndex + 1));
  }

  Value __stdcall *CreateOr(Value *LHS, Value *RHS,
                            const char *Name = "") override {
    return IRBuilder<>::CreateOr(LHS, RHS, Twine(Name ? Name : ""));
  }

  /*Value __stdcall *CreateOr(Value *LHS, const APInt &RHS,
                            const char *Name = "") override {
    return IRBuilder<>::CreateOr(LHS, RHS, Twine(Name ? Name : ""));
  }*/

  Value __stdcall *CreateOr(Value *LHS, uint64_t RHS,
                            const char *Name = "") override {
    return IRBuilder<>::CreateOr(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateOr(Value **Ops, int OpsHiIndex) override {
    return IRBuilder<>::CreateOr(ArrayRef<Value *>(Ops, OpsHiIndex + 1));
  }

  Value __stdcall *CreateXor(Value *LHS, Value *RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateXor(LHS, RHS, Twine(Name ? Name : ""));
  }

  /*Value __stdcall *CreateXor(Value *LHS, const APInt &RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateXor(LHS, RHS, Twine(Name ? Name : ""));
  }*/

  Value __stdcall *CreateXor(Value *LHS, uint64_t RHS,
                             const char *Name = "") override {
    return IRBuilder<>::CreateXor(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFAdd(Value *L, Value *R, const char *Name = "",
                              MDNode *FPMD = nullptr) override {
    return IRBuilder<>::CreateFAdd(L, R, Twine(Name ? Name : ""), FPMD);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFAddFMF(Value *L, Value *R, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFAddFMF(L, R, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFSub(Value *L, Value *R, const char *Name = "",
                              MDNode *FPMD = nullptr) override {
    return IRBuilder<>::CreateFSub(L, R, Twine(Name ? Name : ""), FPMD);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFSubFMF(Value *L, Value *R, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFSubFMF(L, R, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFMul(Value *L, Value *R, const char *Name = "",
                              MDNode *FPMD = nullptr) override {
    return IRBuilder<>::CreateFMul(L, R, Twine(Name ? Name : ""), FPMD);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFMulFMF(Value *L, Value *R, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFMulFMF(L, R, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFDiv(Value *L, Value *R, const char *Name = "",
                              MDNode *FPMD = nullptr) override {
    return IRBuilder<>::CreateFDiv(L, R, Twine(Name ? Name : ""), FPMD);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFDivFMF(Value *L, Value *R, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFDivFMF(L, R, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFRem(Value *L, Value *R, const char *Name = "",
                              MDNode *FPMD = nullptr) override {
    return IRBuilder<>::CreateFRem(L, R, Twine(Name ? Name : ""), FPMD);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFRemFMF(Value *L, Value *R, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFRemFMF(L, R, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateBinOp(Instruction::BinaryOps Opc, Value *LHS,
                               Value *RHS, const char *Name = "",
                               MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateBinOp(Opc, LHS, RHS, Twine(Name ? Name : ""),
                                    FPMathTag);
  }

  CallInst __stdcall *CreateConstrainedFPBinOp(
      Intrinsic::ID ID, Value *L, Value *R, Instruction *FMFSource = nullptr,
      const char *Name = "", MDNode *FPMathTag = nullptr,
      Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None) override {
    return IRBuilder<>::CreateConstrainedFPBinOp(ID, L, R, FMFSource,
                                                 Twine(Name ? Name : ""),
                                                 FPMathTag, Rounding, Except);
  }

  Value __stdcall *CreateNeg(Value *V, const char *Name = "",
                             bool HasNUW = false,
                             bool HasNSW = false) override {
    return IRBuilder<>::CreateNeg(V, Twine(Name ? Name : ""), HasNUW, HasNSW);
  }

  Value __stdcall *CreateNSWNeg(Value *V, const char *Name = "") override {
    return IRBuilder<>::CreateNSWNeg(V, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateNUWNeg(Value *V, const char *Name = "") override {
    return IRBuilder<>::CreateNSWNeg(V, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFNeg(Value *V, const char *Name = "",
                              MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFNeg(V, Twine(Name ? Name : ""), FPMathTag);
  }

  /// Copy fast-math-flags from an instruction rather than using the builder's
  /// default FMF.
  Value __stdcall *CreateFNegFMF(Value *V, Instruction *FMFSource,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFNegFMF(V, FMFSource, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateNot(Value *V, const char *Name = "") override {
    return IRBuilder<>::CreateNot(V, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateUnOp(Instruction::UnaryOps Opc, Value *V,
                              const char *Name = "",
                              MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateUnOp(Opc, V, Twine(Name ? Name : ""), FPMathTag);
  }

  /// Create either a UnaryOperator or BinaryOperator depending on \p Opc.
  /// Correct number of operands must be passed accordingly.
  Value __stdcall *CreateNAryOp(unsigned Opc, Value **Ops, int OpsHiIndex,
                                const char *Name = "",
                                MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateNAryOp(Opc,
                                     ArrayRef<Value *>(Ops, OpsHiIndex + 1),
                                     Twine(Name ? Name : ""), FPMathTag);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//

  AllocaInst __stdcall *CreateAlloca(Type *Ty, unsigned AddrSpace,
                                     Value *ArraySize = nullptr,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateAlloca(Ty, AddrSpace, ArraySize,
                                     Twine(Name ? Name : ""));
  }

  AllocaInst __stdcall *CreateAlloca(Type *Ty, Value *ArraySize = nullptr,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateAlloca(Ty, ArraySize, Twine(Name ? Name : ""));
  }

  LoadInst __stdcall *CreateLoad(Type *Ty, Value *Ptr, bool isVolatile,
                                 MaybeAlign Align,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateAlignedLoad(Ty, Ptr, Align, isVolatile,
                                          Twine(Name ? Name : ""));
  }

  StoreInst __stdcall *CreateStore(Value *Val, Value *Ptr, MaybeAlign Align,
                                   bool isVolatile = false) override {
    return IRBuilder<>::CreateAlignedStore(Val, Ptr, Align, isVolatile);
  };

  FenceInst __stdcall *CreateFence(AtomicOrdering Ordering,
                                   SyncScope::ID SSID = SyncScope::System,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateFence(Ordering, SSID, Twine(Name ? Name : ""));
  }

  AtomicCmpXchgInst __stdcall *
  CreateAtomicCmpXchg(Value *Ptr, Value *Cmp, Value *New,
                      AtomicOrdering SuccessOrdering,
                      AtomicOrdering FailureOrdering,
                      SyncScope::ID SSID = SyncScope::System) override {
    return IRBuilder<>::CreateAtomicCmpXchg(Ptr, Cmp, New, SuccessOrdering,
                                            FailureOrdering, SSID);
  }

  AtomicRMWInst __stdcall *
  CreateAtomicRMW(AtomicRMWInst::BinOp Op, Value *Ptr, Value *Val,
                  AtomicOrdering Ordering,
                  SyncScope::ID SSID = SyncScope::System) override {
    return IRBuilder<>::CreateAtomicRMW(Op, Ptr, Val, Ordering, SSID);
  }

  Value __stdcall *CreateGEP(Value *Ptr, Value **IdxList, int IdxListHiIndex,
                             const char *Name = "") override {
    return IRBuilder<>::CreateGEP(
        Ptr, ArrayRef<Value *>(IdxList, IdxListHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateGEP(Type *Ty, Value *Ptr, Value **IdxList,
                             int IdxListHiIndex,
                             const char *Name = "") override {
    return IRBuilder<>::CreateGEP(
        Ty, Ptr, ArrayRef<Value *>(IdxList, IdxListHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInBoundsGEP(Value *Ptr, Value **IdxList,
                                     int IdxListHiIndex,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInBoundsGEP(
        Ptr, ArrayRef<Value *>(IdxList, IdxListHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInBoundsGEP(Type *Ty, Value *Ptr, Value **IdxList,
                                     int IdxListHiIndex,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInBoundsGEP(
        Ty, Ptr, ArrayRef<Value *>(IdxList, IdxListHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateGEP(Value *Ptr, Value *Idx,
                             const char *Name = "") override {
    return IRBuilder<>::CreateGEP(Ptr, Idx, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateGEP(Type *Ty, Value *Ptr, Value *Idx,
                             const char *Name = "") override {
    return IRBuilder<>::CreateGEP(Ty, Ptr, Idx, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInBoundsGEP(Type *Ty, Value *Ptr, Value *Idx,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInBoundsGEP(Ty, Ptr, Idx,
                                          Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstGEP1_32(Type *Ty, Value *Ptr, unsigned Idx0,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateConstGEP1_32(Ty, Ptr, Idx0,
                                           Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstInBoundsGEP1_32(Type *Ty, Value *Ptr,
                                              unsigned Idx0,
                                              const char *Name = "") override {
    return IRBuilder<>::CreateConstInBoundsGEP1_32(Ty, Ptr, Idx0,
                                                   Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstGEP2_32(Type *Ty, Value *Ptr, unsigned Idx0,
                                      unsigned Idx1,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateConstGEP2_32(Ty, Ptr, Idx0, Idx1,
                                           Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstInBoundsGEP2_32(Type *Ty, Value *Ptr,
                                              unsigned Idx0, unsigned Idx1,
                                              const char *Name = "") override {
    return IRBuilder<>::CreateConstInBoundsGEP2_32(Ty, Ptr, Idx0, Idx1,
                                                   Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstGEP1_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateConstGEP1_64(Ty, Ptr, Idx0,
                                           Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstInBoundsGEP1_64(Type *Ty, Value *Ptr,
                                              uint64_t Idx0,
                                              const char *Name = "") override {
    return IRBuilder<>::CreateConstInBoundsGEP1_64(Ty, Ptr, Idx0,
                                                   Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstGEP2_64(Type *Ty, Value *Ptr, uint64_t Idx0,
                                      uint64_t Idx1,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateConstGEP2_64(Ty, Ptr, Idx0, Idx1,
                                           Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateConstInBoundsGEP2_64(Type *Ty, Value *Ptr,
                                              uint64_t Idx0, uint64_t Idx1,
                                              const char *Name = "") override {
    return IRBuilder<>::CreateConstInBoundsGEP2_64(Ty, Ptr, Idx0, Idx1,
                                                   Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateStructGEP(Type *Ty, Value *Ptr, unsigned Idx,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateConstInBoundsGEP2_32(Ty, Ptr, 0, Idx, Name);
  }

  /// Same as CreateGlobalString, but return a pointer with "i8*" type
  /// instead of a pointer to array of i8.
  ///
  /// If no module is given via \p M, it is take from the insertion point basic
  /// block.
  /*Constant __stdcall *CreateGlobalStringPtr(StringRef Str,
                                            const char *Name = "",
                                            unsigned AddressSpace = 0,
                                            Module *M = nullptr) override {
    return IRBuilder<>::CreateGlobalStringPtr(Str, Twine(Name ? Name : ""),
                                              AddressSpace, M);
  }*/

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Value __stdcall *CreateTrunc(Value *V, Type *DestTy,
                               const char *Name = "") override {
    return IRBuilder<>::CreateTrunc(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateZExt(Value *V, Type *DestTy,
                              const char *Name = "") override {
    return IRBuilder<>::CreateZExt(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSExt(Value *V, Type *DestTy,
                              const char *Name = "") override {
    return IRBuilder<>::CreateSExt(V, DestTy, Twine(Name ? Name : ""));
  }

  /// Create a ZExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value __stdcall *CreateZExtOrTrunc(Value *V, Type *DestTy,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateZExtOrTrunc(V, DestTy, Twine(Name ? Name : ""));
  }

  /// Create a SExt or Trunc from the integer value V to DestTy. Return
  /// the value untouched if the type of V is already DestTy.
  Value __stdcall *CreateSExtOrTrunc(Value *V, Type *DestTy,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateSExtOrTrunc(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFPToUI(Value *V, Type *DestTy,
                                const char *Name = "") override {
    return IRBuilder<>::CreateFPToUI(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFPToSI(Value *V, Type *DestTy,
                                const char *Name = "") override {
    return IRBuilder<>::CreateFPToSI(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateUIToFP(Value *V, Type *DestTy,
                                const char *Name = "") override {
    return IRBuilder<>::CreateUIToFP(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSIToFP(Value *V, Type *DestTy,
                                const char *Name = "") override {
    return IRBuilder<>::CreateSIToFP(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFPTrunc(Value *V, Type *DestTy,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateFPTrunc(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFPExt(Value *V, Type *DestTy,
                               const char *Name = "") override {
    return IRBuilder<>::CreateFPExt(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreatePtrToInt(Value *V, Type *DestTy,
                                  const char *Name = "") override {
    return IRBuilder<>::CreatePtrToInt(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateIntToPtr(Value *V, Type *DestTy,
                                  const char *Name = "") override {
    return IRBuilder<>::CreateIntToPtr(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateBitCast(Value *V, Type *DestTy,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateBitCast(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateAddrSpaceCast(Value *V, Type *DestTy,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateAddrSpaceCast(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateZExtOrBitCast(Value *V, Type *DestTy,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateZExtOrBitCast(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateSExtOrBitCast(Value *V, Type *DestTy,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateSExtOrBitCast(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateTruncOrBitCast(Value *V, Type *DestTy,
                                        const char *Name = "") override {
    return IRBuilder<>::CreateTruncOrBitCast(V, DestTy,
                                             Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateCast(Instruction::CastOps Op, Value *V, Type *DestTy,
                              const char *Name = "") override {
    return IRBuilder<>::CreateCast(Op, V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreatePointerCast(Value *V, Type *DestTy,
                                     const char *Name = "") override {
    return IRBuilder<>::CreatePointerCast(V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *
  CreatePointerBitCastOrAddrSpaceCast(Value *V, Type *DestTy,
                                      const char *Name = "") override {
    return IRBuilder<>::CreatePointerBitCastOrAddrSpaceCast(
        V, DestTy, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateIntCast(Value *V, Type *DestTy, bool isSigned,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateIntCast(V, DestTy, isSigned,
                                      Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateBitOrPointerCast(Value *V, Type *DestTy,
                                          const char *Name = "") override {
    return IRBuilder<>::CreateBitOrPointerCast(V, DestTy,
                                               Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFPCast(Value *V, Type *DestTy,
                                const char *Name = "") override {
    return IRBuilder<>::CreateFPCast(V, DestTy, Twine(Name ? Name : ""));
  }

  CallInst __stdcall *CreateConstrainedFPCast(
      Intrinsic::ID ID, Value *V, Type *DestTy,
      Instruction *FMFSource = nullptr, const char *Name = "",
      MDNode *FPMathTag = nullptr, Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None) override {
    return IRBuilder<>::CreateConstrainedFPCast(ID, V, DestTy, FMFSource,
                                                Twine(Name ? Name : ""),
                                                FPMathTag, Rounding, Except);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//

  Value __stdcall *CreateICmpEQ(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateICmpEQ(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpNE(Value *LHS, Value *RHS,
                                const char *Name = "") override {
    return IRBuilder<>::CreateICmpNE(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpUGT(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpUGT(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpUGE(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpUGE(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpULT(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpULT(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpULE(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpULE(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpSGT(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpSGT(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpSGE(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpSGE(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpSLT(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpSLT(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateICmpSLE(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreateICmpSLE(LHS, RHS, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFCmpOEQ(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpOEQ(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpOGT(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpOGT(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpOGE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpOGE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpOLT(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpOLT(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpOLE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpOLE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpONE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpONE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpORD(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpORD(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpUNO(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpUNO(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpUEQ(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpUEQ(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpUGT(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpUGT(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpUGE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpUGE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpULT(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpULT(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpULE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpULE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateFCmpUNE(Value *LHS, Value *RHS, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpUNE(LHS, RHS, Twine(Name ? Name : ""),
                                      FPMathTag);
  }

  Value __stdcall *CreateICmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                              const char *Name = "") override {
    return IRBuilder<>::CreateICmp(P, LHS, RHS, Twine(Name ? Name : ""));
  }

  // Create a quiet floating-point comparison (i.e. one that raises an FP
  // exception only in the case where an input is a signaling NaN).
  // Note that this differs from CreateFCmpS only if IsFPConstrained is true.
  Value __stdcall *CreateFCmp(CmpInst::Predicate P, Value *LHS, Value *RHS,
                              const char *Name = "",
                              MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmp(P, LHS, RHS, Twine(Name ? Name : ""),
                                   FPMathTag);
  }

  Value __stdcall *CreateCmp(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                             const char *Name = "",
                             MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateCmp(Pred, LHS, RHS, Twine(Name ? Name : ""),
                                  FPMathTag);
  }

  // Create a signaling floating-point comparison (i.e. one that raises an FP
  // exception whenever an input is any NaN, signaling or quiet).
  // Note that this differs from CreateFCmp only if IsFPConstrained is true.
  Value __stdcall *CreateFCmpS(CmpInst::Predicate P, Value *LHS, Value *RHS,
                               const char *Name = "",
                               MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateFCmpS(P, LHS, RHS, Twine(Name ? Name : ""),
                                    FPMathTag);
  }

  CallInst __stdcall *CreateConstrainedFPCmp(
      Intrinsic::ID ID, CmpInst::Predicate P, Value *L, Value *R,
      const char *Name = "",
      Optional<fp::ExceptionBehavior> Except = None) override {
    return IRBuilder<>::CreateConstrainedFPCmp(ID, P, L, R,
                                               Twine(Name ? Name : ""), Except);
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  PHINode __stdcall *CreatePHI(Type *Ty, unsigned NumReservedValues,
                               const char *Name = "") override {
    return IRBuilder<>::CreatePHI(Ty, NumReservedValues,
                                  Twine(Name ? Name : ""));
  }

  CallInst __stdcall *CreateCall(FunctionType *FTy, Value *Callee, Value **Args,
                                 int ArgsHiIndex, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateCall(FTy, Callee,
                                   ArrayRef<Value *>(Args, ArgsHiIndex + 1),
                                   Twine(Name ? Name : ""), FPMathTag);
  }

  CallInst __stdcall *CreateCall(FunctionType *FTy, Value *Callee, Value **Args,
                                 int ArgsHiIndex, OperandBundleDef *OpBundles,
                                 int OpBundlesHiIndex, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateCall(
        FTy, Callee, ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""), FPMathTag);
  }

  CallInst __stdcall *CreateCall(FunctionCallee Callee, Value **Args,
                                 int ArgsHiIndex, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateCall(Callee,
                                   ArrayRef<Value *>(Args, ArgsHiIndex + 1),
                                   Twine(Name ? Name : ""), FPMathTag);
  }

  CallInst __stdcall *CreateCall(FunctionCallee Callee, Value **Args,
                                 int ArgsHiIndex, OperandBundleDef *OpBundles,
                                 int OpBundlesHiIndex, const char *Name = "",
                                 MDNode *FPMathTag = nullptr) override {
    return IRBuilder<>::CreateCall(
        Callee, ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        ArrayRef<OperandBundleDef>(OpBundles, OpBundlesHiIndex + 1),
        Twine(Name ? Name : ""), FPMathTag);
  }

  CallInst __stdcall *CreateConstrainedFPCall(
      Function *Callee, Value **Args, int ArgsHiIndex, const char *Name = "",
      Optional<RoundingMode> Rounding = None,
      Optional<fp::ExceptionBehavior> Except = None) override {
    return IRBuilder<>::CreateConstrainedFPCall(
        Callee, ArrayRef<Value *>(Args, ArgsHiIndex + 1),
        Twine(Name ? Name : ""), Rounding, Except);
  }

  Value __stdcall *CreateSelect(Value *C, Value *True, Value *False,
                                const char *Name = "",
                                Instruction *MDFrom = nullptr) override {
    return IRBuilder<>::CreateSelect(C, True, False, Twine(Name ? Name : ""),
                                     MDFrom);
  }

  VAArgInst __stdcall *CreateVAArg(Value *List, Type *Ty,
                                   const char *Name = "") override {
    return IRBuilder<>::CreateVAArg(List, Ty, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateExtractElement(Value *Vec, Value *Idx,
                                        const char *Name = "") override {
    return IRBuilder<>::CreateExtractElement(Vec, Idx, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateExtractElement(Value *Vec, uint64_t Idx,
                                        const char *Name = "") override {
    return IRBuilder<>::CreateExtractElement(Vec, Idx, Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateInsertElement(Vec, NewElt, Idx,
                                            Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInsertElement(Value *Vec, Value *NewElt, uint64_t Idx,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateInsertElement(Vec, NewElt, Idx,
                                            Twine(Name ? Name : ""));
  }

  /*Value __stdcall *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                                       const char *Name = "") override {
    SmallVector<int, 16> IntMask;
    ShuffleVectorInst::getShuffleMask(cast<Constant>(Mask), IntMask);
    return IRBuilder<>::CreateShuffleVector(V1, V2, IntMask,
                                            Twine(Name ? Name : ""));
  }*/

  /// See class ShuffleVectorInst for a description of the mask representation.
  Value __stdcall *CreateShuffleVector(Value *V1, Value *V2, int *Mask,
                                       int MaxHiIndex,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateShuffleVector(
        V1, V2, ArrayRef<int>(Mask, MaxHiIndex + 1), Twine(Name ? Name : ""));
  }

  /// Create a unary shuffle. The second vector operand of the IR instruction
  /// is poison.
  Value __stdcall *CreateShuffleVector(Value *V, int *Mask, int MaxHiIndex,
                                       const char *Name = "") override {
    return IRBuilder<>::CreateShuffleVector(
        V, ArrayRef<int>(Mask, MaxHiIndex + 1), Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateExtractValue(Value *Agg, unsigned *Idxs,
                                      int IdxsHiIndex,
                                      const char *Name = "") override {
    return IRBuilder<>::CreateExtractValue(
        Agg, ArrayRef<unsigned>(Idxs, IdxsHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateInsertValue(Value *Agg, Value *Val, unsigned *Idxs,
                                     int IdxsHiIndex,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateInsertValue(
        Agg, Val, ArrayRef<unsigned>(Idxs, IdxsHiIndex + 1),
        Twine(Name ? Name : ""));
  }

  LandingPadInst __stdcall *CreateLandingPad(Type *Ty, unsigned NumClauses,
                                             const char *Name = "") override {
    return IRBuilder<>::CreateLandingPad(Ty, NumClauses,
                                         Twine(Name ? Name : ""));
  }

  Value __stdcall *CreateFreeze(Value *V, const char *Name = "") override {
    return IRBuilder<>::CreateFreeze(V, Twine(Name ? Name : ""));
  }

  //===--------------------------------------------------------------------===//
  // Utility creation methods
  //===--------------------------------------------------------------------===//

  /// Return an i1 value testing if \p Arg is null.
  Value __stdcall *CreateIsNull(Value *Arg, const char *Name = "") override {
    return IRBuilder<>::CreateIsNull(Arg, Twine(Name ? Name : ""));
  }

  /// Return an i1 value testing if \p Arg is not null.
  Value __stdcall *CreateIsNotNull(Value *Arg, const char *Name = "") override {
    return IRBuilder<>::CreateIsNotNull(Arg, Twine(Name ? Name : ""));
  }

  /// Return the i64 difference between two pointer values, dividing out
  /// the size of the pointed-to objects.
  ///
  /// This is intended to implement C-style pointer subtraction. As such, the
  /// pointers must be appropriately aligned for their element types and
  /// pointing into the same object.
  Value __stdcall *CreatePtrDiff(Value *LHS, Value *RHS,
                                 const char *Name = "") override {
    return IRBuilder<>::CreatePtrDiff(LHS, RHS, Twine(Name ? Name : ""));
  }

  /// Create a launder.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  Value __stdcall *CreateLaunderInvariantGroup(Value *Ptr) override {
    return IRBuilder<>::CreateLaunderInvariantGroup(Ptr);
  }

  /// \brief Create a strip.invariant.group intrinsic call. If Ptr type is
  /// different from pointer to i8, it's casted to pointer to i8 in the same
  /// address space before call and casted back to Ptr type after call.
  Value __stdcall *CreateStripInvariantGroup(Value *Ptr) override {
    return IRBuilder<>::CreateStripInvariantGroup(Ptr);
  }

  /// Return a vector value that contains \arg V broadcasted to \p
  /// NumElts elements.
  Value __stdcall *CreateVectorSplat(unsigned NumElts, Value *V,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateVectorSplat(NumElts, V, Twine(Name ? Name : ""));
  }

  /// Return a vector value that contains \arg V broadcasted to \p
  /// EC elements.
  Value __stdcall *CreateVectorSplat(ElementCount EC, Value *V,
                                     const char *Name = "") override {
    return IRBuilder<>::CreateVectorSplat(EC, V, Twine(Name ? Name : ""));
  }

  /// Return a value that has been extracted from a larger integer type.
  Value __stdcall *CreateExtractInteger(const DataLayout &DL, Value *From,
                                        IntegerType *ExtractedTy,
                                        uint64_t Offset,
                                        const char *Name) override {
    return IRBuilder<>::CreateExtractInteger(DL, From, ExtractedTy, Offset,
                                             Twine(Name ? Name : ""));
  }

  Value __stdcall *CreatePreserveArrayAccessIndex(Type *ElTy, Value *Base,
                                                  unsigned Dimension,
                                                  unsigned LastIndex,
                                                  MDNode *DbgInfo) override {
    return IRBuilder<>::CreatePreserveArrayAccessIndex(ElTy, Base, Dimension,
                                                       LastIndex, DbgInfo);
  }

  Value __stdcall *CreatePreserveUnionAccessIndex(Value *Base,
                                                  unsigned FieldIndex,
                                                  MDNode *DbgInfo) override {
    return IRBuilder<>::CreatePreserveUnionAccessIndex(Base, FieldIndex,
                                                       DbgInfo);
  }

  Value __stdcall *CreatePreserveStructAccessIndex(Type *ElTy, Value *Base,
                                                   unsigned Index,
                                                   unsigned FieldIndex,
                                                   MDNode *DbgInfo) override {
    return IRBuilder<>::CreatePreserveStructAccessIndex(ElTy, Base, Index,
                                                        FieldIndex, DbgInfo);
  }

  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  CallInst __stdcall *
  CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                            unsigned Alignment,
                            Value *OffsetValue = nullptr) override {
    return IRBuilder<>::CreateAlignmentAssumption(DL, PtrValue, Alignment,
                                                  OffsetValue);
  }

  /// Create an assume intrinsic call that represents an alignment
  /// assumption on the provided pointer.
  ///
  /// An optional offset can be provided, and if it is provided, the offset
  /// must be subtracted from the provided pointer to get the pointer with the
  /// specified alignment.
  ///
  /// This overload handles the condition where the Alignment is dependent
  /// on an existing value rather than a static value.
  CallInst __stdcall *
  CreateAlignmentAssumption(const DataLayout &DL, Value *PtrValue,
                            Value *Alignment,
                            Value *OffsetValue = nullptr) override {
    return IRBuilder<>::CreateAlignmentAssumption(DL, PtrValue, Alignment,
                                                  OffsetValue);
  }
};

void Adapter_ISequentialStream::write_impl(const char *Ptr, size_t Size) {
  real->Write(Ptr, Size, nullptr);
}

Adapter_ISequentialStream::Adapter_ISequentialStream(
    ISequentialStream *stream) {
  real = stream;
  real->AddRef();
}

Adapter_ISequentialStream::~Adapter_ISequentialStream() {
  ISequentialStream *old = real;
  real = nullptr;
  old->Release();
}

void Adapter_IStream::pwrite_impl(const char *Ptr, size_t Size,
                                 uint64_t Offset) {
  LARGE_INTEGER ofs;
  ofs.QuadPart = Offset;
  real->Seek(ofs, STREAM_SEEK_SET, nullptr);
  real->Write(Ptr, Size, nullptr);
}

void Adapter_IStream::write_impl(const char *Ptr, size_t Size) {
  real->Write(Ptr, Size, nullptr);
}

uint64_t Adapter_IStream::current_pos() const {
  ULARGE_INTEGER pos;
  LARGE_INTEGER ofs = {};
  real->Seek(ofs, STREAM_SEEK_CUR, &pos);
  return pos.QuadPart;
}

Adapter_IStream::Adapter_IStream(
    IStream *stream) {
  real = stream;
  real->AddRef();
}

Adapter_IStream::~Adapter_IStream() {
  IStream *old = real;
  real = nullptr;
  old->Release();
}

class Adapter_ModulePass : public ModulePass {
private:
  ILLVMModulePass *real;

public:
  Adapter_ModulePass(ILLVMModulePass *pass)
      : ModulePass(*pass->GetPassID()), real(pass) {
    real->AddRef();
  }

  // Force out-of-line virtual method.
  ~Adapter_ModulePass() override {
    ILLVMModulePass *old = real;
    real = nullptr;
    old->Release();
  }

  /// runOnModule - Virtual method overriden by subclasses to process the module
  /// being operated on.
  bool runOnModule(Module &M) override { return false; }
};

class ImplLLVMModule : public Module, public ILLVMModule {
public:
  ImplLLVMModule(StringRef ModuleID, LLVMContext &C) : Module(ModuleID, C) {}

  void __stdcall SetDataLayout(const char *dataLayout) override {
    setDataLayout(StringRef(dataLayout));
  }

  void __stdcall SetDataLayout(DataLayout *dataLayout) override {
    setDataLayout(*dataLayout);
  }

  const DataLayout __stdcall *GetDataLayout() override {
    return &getDataLayout();
  }

  void __stdcall SetTargetTriple(const char *triple) override {
    setTargetTriple(StringRef(triple));
  }

  Function __stdcall *CreateFunction(FunctionType *Ty,
                                     GlobalValue::LinkageTypes Linkage,
                                     unsigned AddrSpace,
                                     const char *N) override {
    return Function::Create(Ty, Linkage, AddrSpace, Twine(N ? N : ""), this);
  }

  virtual void __stdcall WriteBitcodeToFile(ISequentialStream *out) override {
    Adapter_ISequentialStream stream(out);
    llvm::WriteBitcodeToFile(*this, stream);
    stream.flush();
  }

  virtual bool __stdcall Verify(ISequentialStream *out) override {
    bool result = false;
    if (out == nullptr)
      result = verifyModule(*this);
    else {
      Adapter_ISequentialStream stream(out);
      result = verifyModule(*this, &stream, nullptr);
      stream.flush();
    }
    return result;
  }
};

ILLVMModule __stdcall *ImplLLVMContext::CreateModule(const char *moduleID) {
  return new ImplLLVMModule(StringRef(moduleID), *this);
}

void __stdcall ImplLLVMContext::CreateBuilder(IIRBuilder **builder) {
  *builder = new ImplIRBuilder(this);
  (*builder)->AddRef();
}

class ImplLLVMPassManager : public legacy::PassManager,
                            public ILLVMPassManager,
                            public ImplUnknown {
public:
  HRESULT __stdcall QueryInterface(const GUID &riid,
                                   void **ppvObject) override {
    return ImplUnknown::QueryInterface(riid, ppvObject);
  }
  ULONG __stdcall AddRef(void) override { return ImplUnknown::AddRef(); }
  ULONG __stdcall Release(void) override { return ImplUnknown::Release(); }

  ImplLLVMPassManager() {}
  virtual ~ImplLLVMPassManager() {}

  void __stdcall AddPass(Pass *pass) override {
    legacy::PassManager::add(pass);
  }

  void __stdcall AddPass(ILLVMPassInterface *pass) override {
    Pass *temp;
    ILLVMModulePass *mpass;
    if (!pass->QueryInterface(&mpass))
      temp = new Adapter_ModulePass(mpass);
    else
      return;
    legacy::PassManager::add(temp);
  }

  bool __stdcall Run(ILLVMModule *M) override {
    return legacy::PassManager::run(*dynamic_cast<ImplLLVMModule *>(M));
  }
};

void createPassManager(ILLVMPassManager **ppvContext) {
  *ppvContext = new ImplLLVMPassManager();
  (*ppvContext)->AddRef();
}

void createContext(ILLVMContext **ppvContext) {
  *ppvContext = new ImplLLVMContext();
  (*ppvContext)->AddRef();
}

Function *Function_Create(FunctionType *Ty, GlobalValue::LinkageTypes Linkage,
                          unsigned AddrSpace, const char *N) {
  return Function::Create(Ty, Linkage, AddrSpace, Twine(N ? N : ""));
}

void FunctionSetCallingConversion(Function *func, CallingConv::ID CC) {
  func->setCallingConv(CC);
}

Argument *FunctionGetArgument(Function *func, unsigned int index) {
  if (index >= func->arg_size())
    return nullptr;
  return func->getArg(index);
}

unsigned int FunctionArgumentCount(Function *func) { return func->arg_size(); }

void ValueSetName(Value *arg, const char *name) {
  arg->setName(Twine(name ? name : ""));
}

size_t ValueGetName(Value *arg, char *buffer, size_t bufferSize) {
  auto name = arg->getName();
  if (buffer && name.size() <= bufferSize) {

    memcpy(buffer, name.data(), name.size());
    buffer[name.size()] = '\0';
  }
  return name.size();
}

BasicBlock *FunctionCreateBasicBlock(Function *parent, const char *name,
                                     BasicBlock *insertBefore) {
  return BasicBlock::Create(parent->getContext(), Twine(name ? name : ""),
                            parent, insertBefore);
}

AllocaInst *AllocaInst_CreateBefore(Type *type, unsigned addrSpace,
                                    Value *arraySize, Align align,
                                    const char *name,
                                    Instruction *insertBefore) {
  return new AllocaInst(type, addrSpace, arraySize, align,
                        Twine(name ? name : ""), insertBefore);
}

AllocaInst *AllocaInst_CreateDefaultBefore(Type *type, Value *arraySize,
                                           const char *name,
                                           Instruction *insertBefore) {
  return new AllocaInst(
      type, insertBefore->getModule()->getDataLayout().getAllocaAddrSpace(),
      arraySize, Twine(name ? name : ""), insertBefore);
}

AllocaInst *AllocaInst_CreateAtEnd(Type *type, unsigned addrSpace,
                                   Value *arraySize, Align align,
                                   const char *name, BasicBlock *insertAtEnd) {
  return new AllocaInst(type, addrSpace, arraySize, align,
                        Twine(name ? name : ""), insertAtEnd);
}

AllocaInst *AllocaInst_CreateDefaultAtEnd(Type *type, Value *arraySize,
                                          const char *name,
                                          BasicBlock *insertAtEnd) {
  return new AllocaInst(
      type, insertAtEnd->getModule()->getDataLayout().getAllocaAddrSpace(),
      arraySize, Twine(name ? name : ""), insertAtEnd);
}

void AllocaInstSetAlignment(AllocaInst *inst, Align align) {
  inst->setAlignment(align);
}

StoreInst *StoreInst_CreateBefore(Value *Val, Value *Ptr, bool isVolatile,
                                  Align align, AtomicOrdering Order,
                                  SyncScope::ID SSID,
                                  Instruction *InsertBefore) {
  return new StoreInst(Val, Ptr, isVolatile, align, Order, SSID, InsertBefore);
}

StoreInst *StoreInst_CreateDefaultBefore(Value *Val, Value *Ptr,
                                         bool isVolatile,
                                         Instruction *InsertBefore) {
  return new StoreInst(Val, Ptr, isVolatile, InsertBefore);
}

StoreInst *StoreInst_CreateAtEnd(Value *Val, Value *Ptr, bool isVolatile,
                                 Align align, AtomicOrdering Order,
                                 SyncScope::ID SSID, BasicBlock *InsertAtEnd) {
  return new StoreInst(Val, Ptr, isVolatile, align, Order, SSID, InsertAtEnd);
}

StoreInst *StoreInst_CreateDefaultAtEnd(Value *Val, Value *Ptr, bool isVolatile,
                                        BasicBlock *InsertAtEnd) {
  return new StoreInst(Val, Ptr, isVolatile, InsertAtEnd);
}

void StoreInstSetAlignment(llvm::StoreInst *inst, Align align) {
  inst->setAlignment(align);
}

LoadInst *LoadInst_CreateBefore(Type *Ty, Value *Ptr, const char *NameStr,
                                bool isVolatile, Align Align,
                                AtomicOrdering Order, SyncScope::ID SSID,
                                Instruction *InsertBefore) {
  return new LoadInst(Ty, Ptr, Twine(NameStr ? NameStr : ""), isVolatile, Align,
                      Order, SSID, InsertBefore);
}

LoadInst *LoadInst_CreateDefaultBefore(Type *Ty, Value *Ptr,
                                       const char *NameStr, bool isVolatile,
                                       Instruction *InsertBefore) {
  return new LoadInst(Ty, Ptr, Twine(NameStr ? NameStr : ""), isVolatile,
                      InsertBefore);
}

LoadInst *LoadInst_CreateAtEnd(Type *Ty, Value *Ptr, const char *NameStr,
                               bool isVolatile, Align Align,
                               AtomicOrdering Order, SyncScope::ID SSID,
                               BasicBlock *InsertAtEnd) {
  return new LoadInst(Ty, Ptr, Twine(NameStr ? NameStr : ""), isVolatile, Align,
                      Order, SSID, InsertAtEnd);
}

LoadInst *LoadInst_CreateDefaultAtEnd(Type *Ty, Value *Ptr, const char *NameStr,
                                      bool isVolatile,
                                      BasicBlock *InsertAtEnd) {
  return new LoadInst(Ty, Ptr, Twine(NameStr ? NameStr : ""), isVolatile,
                      InsertAtEnd);
}

void LoadInstSetAlignment(LoadInst *inst, Align align) {
  inst->setAlignment(align);
}

void OperandBundleDef_Constuctor(Value **bundle, int bundleHiIndex, char *Tag,
                                 OperandBundleDef *out) {
  new (out) OperandBundleDef(Tag, ArrayRef<Value *>(bundle, bundleHiIndex + 1));
}

void OperandBundleDef_Destructor(OperandBundleDef *out) {
  out->~OperandBundleDef();
}

bool VerifyFunction(const Function *F, ISequentialStream *out) {
  bool result = false;
  if (out == nullptr)
    result = verifyFunction(*F);
  else {
    Adapter_ISequentialStream stream(out);
    result = verifyFunction(*F, &stream);
    stream.flush();
  }
  return result;
}

/// addPassesToX helper drives creation and initialization of TargetPassConfig.
static TargetPassConfig *
addPassesToGenerateCode(LLVMTargetMachine &TM, PassManagerBase &PM,
                        bool DisableVerify,
                        MachineModuleInfoWrapperPass &MMIWP) {
  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *PassConfig = TM.createPassConfig(PM);
  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);
  PM.add(PassConfig);
  PM.add(&MMIWP);

  if (PassConfig->addISelPasses())
    return nullptr;
  PassConfig->addMachinePasses();
  PassConfig->setInitialized();
  return PassConfig;
}

void InitAll() {
  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeEntryExitInstrumenterPass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
  initializeExpandReductionsPass(*Registry);
  initializeHardwareLoopsPass(*Registry);
  initializeTransformUtils(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);
}

static void compileModule(Module *M, raw_pwrite_stream *OS) {
  codegen::RegisterCodeGenFlags CGF;
  // Load the module to be compiled...
  SMDiagnostic Err;
  Triple TheTriple(M->getTargetTriple());
  std::string CPUStr = codegen::getCPUStr(),
              FeaturesStr = codegen::getFeaturesStr();

  // Set attributes on functions as loaded from MIR from command line arguments.
  auto setMIRFunctionAttributes = [&CPUStr, &FeaturesStr](Function &F) {
    codegen::setFunctionAttributes(CPUStr, FeaturesStr, F);
  };

  auto MAttrs = codegen::getMAttrs();
  bool SkipModule = codegen::getMCPU() == "help" ||
                    (!MAttrs.empty() && MAttrs.front() == "help");

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;

  TargetOptions Options;
  auto InitializeOptions = [&](const Triple &TheTriple) {
    Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
    //Options.DisableIntegratedAS = NoIntegratedAssembler;
    //Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
    //Options.MCOptions.MCUseDwarfDirectory = EnableDwarfDirectory;
    //Options.MCOptions.AsmVerbose = AsmVerbose;
    //Options.MCOptions.PreserveAsmComments = PreserveComments;
    //Options.MCOptions.IASSearchPaths = IncludeDirs;
    //Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
  };

  Optional<Reloc::Model> RM = codegen::getExplicitRelocModel();

  const Target *TheTarget = nullptr;
  std::unique_ptr<TargetMachine> Target;
 {
      std::string Error;
      TheTarget =
          TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);

      InitializeOptions(TheTriple);
      Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
          TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM,
          codegen::getExplicitCodeModel(), OLvl));
      assert(Target && "Could not allocate target machine!");
  };

  assert(M && "Should have exited if we didn't have a module!");
  if (codegen::getFloatABIForCalls() != FloatABI::Default)
    Options.FloatABIType = codegen::getFloatABIForCalls();

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (false)//DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(CPUStr, FeaturesStr, *M);

  {
    LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
    MachineModuleInfoWrapperPass *MMIWP =
        new MachineModuleInfoWrapperPass(&LLVMTM);
    Target->addPassesToEmitFile(PM, *OS, nullptr, CodeGenFileType::CGFT_ObjectFile, true, MMIWP);

    const_cast<TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())
        ->Initialize(MMIWP->getMMI().getContext(), *Target);

    PM.run(*M);
  }
}

TargetOptions DefaultTargetOptions() {
  TargetOptions Options;
  Options.StackAlignmentOverride = 0;
  Options.StackSymbolOrdering = 1;
  Options.EnableGlobalISel = 0;
  Options.GlobalISelAbort = GlobalISelAbortMode::Enable;
  Options.UseInitArray = 1;
  Options.XCOFFTracebackTable = 1;
  Options.UniqueSectionNames = 1;
  Options.TrapUnreachable = 1;
  Options.EmitStackSizeSection = 0;
  Options.BBSections = BasicBlockSection::None;
  Options.EmitCallSiteInfo = 0;
  Options.SupportsDebugEntryValues = 0;
  Options.StackProtectorGuardOffset = 0xffffffff;
  Options.StackProtectorGuard = StackProtectorGuards::None;
  Options.FloatABIType = FloatABI::ABIType::Default;
  Options.AllowFPOpFusion = FPOpFusion::FPOpFusionMode::Standard;
  Options.ThreadModel = ThreadModel::Model::POSIX;
  Options.EABIVersion = EABI::Default;
  Options.DebuggerTuning = DebuggerKind::Default;
  Options.setFPDenormalMode(DenormalMode(DenormalMode::DenormalModeKind::IEEE,
                                         DenormalMode::DenormalModeKind::IEEE));

  
  Options.ExceptionModel = ExceptionHandling::SjLj;
  Options.setFP32DenormalMode(DenormalMode(DenormalMode::DenormalModeKind::IEEE,
                                         DenormalMode::DenormalModeKind::IEEE));
  return Options;
}

bool Generate(ILLVMModule *m, IStream *stream) {
  InitAll();
  ImplLLVMModule *mm = static_cast<ImplLLVMModule *>(m);
  Adapter_IStream Out(stream);

  //compileModule(mm, &Out);
  //
  std::string triple = mm->getTargetTriple();
  auto TheTriple = Triple(triple);
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget("", TheTriple, Error);
  std::string FeaturesStr = "";
  std::string CPUStr = "";//"i486";
  TargetOptions Options = DefaultTargetOptions();
  Options.DataSections = TheTriple.hasDefaultDataSections();
  LLVMTargetMachine *Target =
      static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
          TheTriple.getTriple(), CPUStr , FeaturesStr, Options, None));

  ImplLLVMPassManager PM;
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(TheTriple);

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (false)//DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  //codegen::setFunctionAttributes(CPUStr, FeaturesStr, *mm);

  MachineModuleInfoWrapperPass *MMIWP =
      new MachineModuleInfoWrapperPass(Target);

  //Target->addPassesToEmitFile(PM, Out, nullptr, CGFT_ObjectFile, true, MMIWP);
  #pragma region addPassesToEmitFile
  auto passConfig = addPassesToGenerateCode(*Target, PM, false, *MMIWP);

  #pragma region addAsmPrinter
  //Target->addAsmPrinter(PM, Out, nullptr, CGFT_ObjectFile,
  //                      MMIWP->getMMI().getContext());
  Expected<std::unique_ptr<MCStreamer>> MCStreamerOrErr =
      Target->createMCStreamer(Out, nullptr, CGFT_ObjectFile,
                               MMIWP->getMMI().getContext());
  if (auto Err = MCStreamerOrErr.takeError())
    return true;

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer =
      TheTarget->createAsmPrinter(*Target, std::move(*MCStreamerOrErr));
  if (!Printer)
    return true;

  PM.add(Printer);
  #pragma endregion addAsmPrinter
  PM.add(createFreeMachineFunctionPass());
  #pragma endregion
  const_cast<TargetLoweringObjectFile *>(Target->getObjFileLowering())
      ->Initialize(MMIWP->getMMI().getContext(), *Target);

  bool value = PM.run(*mm);

  Out.flush();

  return value;
}

int GetAlignSize(int t) {
  switch (t) {
  case 0:
    return sizeof(Align);
  case 1:
    return sizeof(MaybeAlign);
  case 2:
    return Align(10).value();
  case 3:
    return sizeof(OperandBundleDef);
  case 4:
    return sizeof(TargetOptions);
  case 5:
    return sizeof(DenormalMode);
  case 6:
    return sizeof(MCTargetOptions);
  default:
    return sizeof(bool);
  }
}