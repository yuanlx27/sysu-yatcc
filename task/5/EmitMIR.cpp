// Lower LLVM IR into RV64 MIR and emit assembly.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/CodeGen/Register.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include "AllocReg.h"
#include "EmitAsm.h"
#include "EmitMIR.h"

#define GET_REGINFO_ENUM
#include "RISCVGenRegisterInfo.inc"
#undef GET_REGINFO_ENUM

#define GET_INSTRINFO_ENUM
#include "RISCVGenInstrInfo.inc"
#undef GET_INSTRINFO_ENUM

namespace {

// 把 LLVM 的寄存器对象包装成 MCOperand，后面拼机器指令时直接复用。
static llvm::MCOperand mcReg(llvm::Register r)
{
  return llvm::MCOperand::createReg(r.id());
}

// 把立即数包装成 MCOperand，和寄存器操作数保持统一接口。
static llvm::MCOperand mcImm(int64_t v)
{
  return llvm::MCOperand::createImm(v);
}

class FuncLowering final {
public:
  // 保存模块信息、数据布局和汇编输出器，作为整个函数翻译阶段的上下文。
  FuncLowering(const llvm::Module& m, RvAsmEmitter& as)
      : mod_(m)
      , dl_(m.getDataLayout())
      , as_(as)
  {
  }

  // 把一个 LLVM 函数整体翻译成 MIR，再交给寄存器分配和汇编输出。
  void emitFunction(const llvm::Function& f)
  {
    if (f.isDeclaration())
      return;
    if (f.getName().starts_with("llvm."))
      return;

    resetForFunction(f);
    computeFrameLayoutBase(f);

    as_.emitDirective("\t.text");
    as_.emitDirective("\t.align 2");
    as_.emitDirective(("\t.globl " + f.getName()).str());
    as_.emitDirective(("\t.type " + f.getName() + ", @function").str());
    as_.emitLabel(f.getName());

    emitPrologue();
    spillIncomingArgs(f);

    LinearScanAllocator ra;
    RvInstEmitter emitter(as_, vregSpill_, localBytes_, saveRaOff_, saveS0Off_);

    for (const llvm::BasicBlock& bb : f) {
      as_.emitLabel(bbLabel(bb));
      insts_.clear();
      pendingEdges_.clear();
      curBB_ = &bb;
      for (const llvm::Instruction& inst : bb) {
        if (llvm::isa<llvm::PHINode>(inst))
          continue;
        emitInst(inst);
      }
      ra.allocate(insts_);
      emitter.emit(insts_, ra.physMap());
      for (const EdgeBlock& edge : pendingEdges_) {
        as_.emitLabel(edge.label);
        insts_.clear();
        emitPhiMoves(edge.moves);
        emitVBranchUncond(edge.target);
        ra.allocate(insts_);
        emitter.emit(insts_, ra.physMap());
      }
    }
  }

private:
  const llvm::Module& mod_;
  const llvm::DataLayout& dl_;
  RvAsmEmitter& as_;

  const llvm::Function* curF_ = nullptr;
  const llvm::BasicBlock* curBB_ = nullptr;
  llvm::DenseMap<const llvm::BasicBlock*, std::string> bbLabel_;
  llvm::DenseMap<const llvm::Value*, llvm::Register> valueVReg_;
  llvm::DenseMap<llvm::Register, int32_t> vregSpill_;
  llvm::DenseMap<const llvm::AllocaInst*, int32_t> allocaOff_;
  int32_t localBytes_ = 0;
  int32_t saveRaOff_ = 0;
  int32_t saveS0Off_ = 0;
  unsigned nextVReg_ = 1;
  unsigned tempSlots_ = 8;
  llvm::SmallVector<llvm::Register, 8> tempVRegs_;
  unsigned tempCursor_ = 0;
  llvm::SmallVector<VInst, 64> insts_;
  unsigned edgeId_ = 0;

  struct PhiMove final {
    llvm::Register dst;
    const llvm::Value* src = nullptr;
  };

  struct EdgeBlock final {
    std::string label;
    std::string target;
    llvm::SmallVector<PhiMove, 4> moves;
  };

  llvm::SmallVector<EdgeBlock, 4> pendingEdges_;

  // 申请一个新的虚拟寄存器，表示“这个值先放在哪里”。
  llvm::Register newVReg()
  {
    return llvm::Register::index2VirtReg(nextVReg_++);
  }

  // 开始处理新函数前重置状态，并给每个基本块分配一个汇编标签。
  void resetForFunction(const llvm::Function& f)
  {
    curF_ = &f;
    bbLabel_.clear();
    valueVReg_.clear();
    vregSpill_.clear();
    allocaOff_.clear();
    nextVReg_ = 1;
    tempVRegs_.clear();
    tempCursor_ = 0;
    insts_.clear();

    unsigned bb_id = 0;
    for (const llvm::BasicBlock& bb : f) {
      bbLabel_[&bb] = (".Lbb." + f.getName() + "." + std::to_string(bb_id++)).str();
    }
  }

  // 查询基本块对应的标签，分支和跳转都会用到它。
  std::string bbLabel(const llvm::BasicBlock& bb) const
  {
    auto it = bbLabel_.find(&bb);
    assert(it != bbLabel_.end());
    return it->second;
  }

  // 预先规划栈帧：给 SSA 值、alloca 对象、临时寄存器和保存现场留空间。
  void computeFrameLayoutBase(const llvm::Function& f)
  {
    int32_t off = 0;

    // Stack slots for SSA values (sp-relative, positive offsets).
    for (const llvm::Argument& arg : f.args()) {
      llvm::Register vreg = newVReg();
      valueVReg_[&arg] = vreg;
      vregSpill_[vreg] = off;
      off += 8;
    }

    for (const llvm::BasicBlock& bb : f) {
      for (const llvm::Instruction& inst : bb) {
        if (inst.getType()->isVoidTy())
          continue;
        llvm::Register vreg = newVReg();
        valueVReg_[&inst] = vreg;
        vregSpill_[vreg] = off;
        off += 8;
      }
    }

    // Stack storage for allocas.
    for (const llvm::BasicBlock& bb : f) {
      for (const llvm::Instruction& inst : bb) {
        const auto* ai = llvm::dyn_cast<llvm::AllocaInst>(&inst);
        if (!ai)
          continue;
        llvm::Type* ty = ai->getAllocatedType();
        if (!ai->isStaticAlloca())
          llvm::report_fatal_error("dynamic alloca is not supported");
        int64_t sz = dl_.getTypeAllocSize(ty);
        int64_t cnt = 1;
        if (const llvm::Value* arr = ai->getArraySize()) {
          if (const auto* ci = llvm::dyn_cast<llvm::ConstantInt>(arr)) {
            cnt = ci->getSExtValue();
          } else {
            llvm::report_fatal_error("non-constant alloca array size");
          }
        }
        int64_t bytes = sz * cnt;
        int64_t align = std::max<int64_t>(8, static_cast<int64_t>(ai->getAlign().value()));
        off = static_cast<int32_t>((off + align - 1) & ~(align - 1));
        allocaOff_[ai] = off;
        off += static_cast<int32_t>(bytes);
      }
    }

    // Temp vreg spill slots.
    for (unsigned i = 0; i < tempSlots_; ++i) {
      llvm::Register vreg = newVReg();
      tempVRegs_.push_back(vreg);
      vregSpill_[vreg] = off;
      off += 8;
    }

    // Reserve space for saved ra/s0 and align to 16 bytes.
    off += 16;
    localBytes_ = static_cast<int32_t>((off + 15) & ~15);
    saveRaOff_ = localBytes_ - 8;
    saveS0Off_ = localBytes_ - 16;
  }

  // 找到某个 LLVM 值对应的虚拟寄存器；这是 emitInst/emitV 之间的桥梁。
  llvm::Register vregOf(const llvm::Value* v) const
  {
    auto it = valueVReg_.find(v);
    if (it == valueVReg_.end())
      llvm::report_fatal_error(llvm::Twine("missing vreg for value: ") + v->getName());
    return it->second;
  }

  // 取一个临时虚拟寄存器，专门给中间计算或常量加载使用。
  llvm::Register nextTempVReg()
  {
    if (tempVRegs_.empty())
      llvm::report_fatal_error("temp vregs not initialized");
    llvm::Register vreg = tempVRegs_[tempCursor_++ % tempVRegs_.size()];
    return vreg;
  }

  // 查询某个 alloca 在当前栈帧里的偏移。
  int32_t allocaObjectOff(const llvm::AllocaInst& ai) const
  {
    auto it = allocaOff_.find(&ai);
    if (it == allocaOff_.end())
      llvm::report_fatal_error("missing stack object for alloca");
    return it->second;
  }

  // 输出函数序言：开栈帧、保存返回地址和帧指针，并建立新的 s0。
  void emitPrologue()
  {
    as_.emitLine("addi sp, sp, -" + std::to_string(localBytes_));
    as_.emitLine("sd ra, " + std::to_string(saveRaOff_) + "(sp)");
    as_.emitLine("sd s0, " + std::to_string(saveS0Off_) + "(sp)");
    as_.emitLine("addi s0, sp, " + std::to_string(localBytes_));
  }

  // 把形参统一写回栈槽，后续就能像普通 SSA 值一样按虚拟寄存器处理。
  void spillIncomingArgs(const llvm::Function& f)
  {
    unsigned idx = 0;
    for (const llvm::Argument& arg : f.args()) {
      int32_t dst = vregSpill_[vregOf(&arg)];
      if (idx < 8) {
        as_.emitLine("sd a" + std::to_string(idx) + ", " + std::to_string(dst) + "(sp)");
      } else {
        int32_t src_off = static_cast<int32_t>((idx - 8) * 8);
        as_.emitLine("ld t0, " + std::to_string(src_off) + "(s0)");
        as_.emitLine("sd t0, " + std::to_string(dst) + "(sp)");
      }
      ++idx;
    }
  }

  // 生成一条最基础的 MC 指令，并放进当前基本块的 MIR 序列。
  void emitMC(unsigned opcode, llvm::SmallVector<llvm::MCOperand, 4> ops, std::string sym = {})
  {
    VInst vi;
    vi.kind = VInst::Kind::kMC;
    vi.inst.setOpcode(opcode);
    for (auto& op : ops)
      vi.inst.addOperand(op);
    vi.sym = std::move(sym);
    insts_.push_back(std::move(vi));
  }

  // 生成“加载立即数到虚拟寄存器”的 MIR。
  void emitVLoadImm(llvm::Register dst, int64_t imm)
  {
    emitMC(llvm::RISCV::ADDI, { mcReg(dst), mcReg(llvm::Register(llvm::RISCV::X0)), mcImm(imm) });
  }

  // 生成“取全局符号地址”的 MIR。
  void emitVLoadAddr(llvm::Register dst, llvm::StringRef sym)
  {
    emitMC(llvm::RISCV::PseudoLA, { mcReg(dst) }, sym.str());
  }

  // 用 addi 0 实现寄存器复制，后面很多类型转换都会复用它。
  void emitVMov(llvm::Register dst, llvm::Register src)
  {
    emitMC(llvm::RISCV::ADDI, { mcReg(dst), mcReg(src), mcImm(0) });
  }

  // 生成加法 MIR。
  void emitVAdd(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::ADD, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成减法 MIR。
  void emitVSub(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SUB, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成乘法 MIR。
  void emitVMul(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::MUL, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成有符号除法 MIR。
  void emitVDiv(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::DIV, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成有符号取余 MIR。
  void emitVRem(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::REM, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成加立即数 MIR，常用于地址计算。
  void emitVAddImm(llvm::Register dst, llvm::Register src, int64_t imm)
  {
    emitMC(llvm::RISCV::ADDI, { mcReg(dst), mcReg(src), mcImm(imm) });
  }

  // 以 sp 为基址做地址计算，通常用于把栈上对象地址写进寄存器。
  void emitVAddImmSp(llvm::Register dst, int64_t imm)
  {
    emitMC(llvm::RISCV::ADDI, { mcReg(dst), mcReg(llvm::Register(llvm::RISCV::X2)), mcImm(imm) });
  }

  // 按 8 字节从内存读取到寄存器。
  void emitVLoad(llvm::Register dst, llvm::Register addr)
  {
    emitMC(llvm::RISCV::LD, { mcReg(dst), mcReg(addr), mcImm(0) });
  }

  // 按 8 字节把寄存器内容写回内存。
  void emitVStore(llvm::Register value, llvm::Register addr)
  {
    emitMC(llvm::RISCV::SD, { mcReg(value), mcReg(addr), mcImm(0) });
  }

  // 按 4 字节读取，给 i32 等类型使用。
  void emitVLoad32(llvm::Register dst, llvm::Register addr)
  {
    emitMC(llvm::RISCV::LW, { mcReg(dst), mcReg(addr), mcImm(0) });
  }

  // 按 4 字节写回，给 i32 等类型使用。
  void emitVStore32(llvm::Register value, llvm::Register addr)
  {
    emitMC(llvm::RISCV::SW, { mcReg(value), mcReg(addr), mcImm(0) });
  }

  // 生成按位异或 MIR。
  void emitVXor(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::XOR, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成异或立即数 MIR，常用来做 0/1 取反。
  void emitVXori(llvm::Register dst, llvm::Register src, int64_t imm)
  {
    emitMC(llvm::RISCV::XORI, { mcReg(dst), mcReg(src), mcImm(imm) });
  }

  // 生成有符号“小于比较” MIR，结果是 0 或 1。
  void emitVSlt(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SLT, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成无符号“小于比较” MIR，结果是 0 或 1。
  void emitVSltu(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SLTU, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成无符号“小于立即数比较” MIR，常用来判断是否等于 0。
  void emitVSltiu(llvm::Register dst, llvm::Register src, int64_t imm)
  {
    emitMC(llvm::RISCV::SLTIU, { mcReg(dst), mcReg(src), mcImm(imm) });
  }

  // 生成按位与 MIR。
  void emitVAnd(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::AND, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成按位或 MIR。
  void emitVOr(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::OR, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成左移 MIR。
  void emitVSll(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SLL, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成逻辑右移 MIR。
  void emitVSrl(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SRL, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成算术右移 MIR。
  void emitVSra(llvm::Register dst, llvm::Register lhs, llvm::Register rhs)
  {
    emitMC(llvm::RISCV::SRA, { mcReg(dst), mcReg(lhs), mcReg(rhs) });
  }

  // 生成“左移立即数” MIR。
  void emitVSlli(llvm::Register dst, llvm::Register lhs, int64_t imm)
  {
    emitMC(llvm::RISCV::SLLI, { mcReg(dst), mcReg(lhs), mcImm(imm) });
  }

  // 生成“逻辑右移立即数” MIR。
  void emitVSrli(llvm::Register dst, llvm::Register lhs, int64_t imm)
  {
    emitMC(llvm::RISCV::SRLI, { mcReg(dst), mcReg(lhs), mcImm(imm) });
  }

  // 生成“算术右移立即数” MIR。
  void emitVSrai(llvm::Register dst, llvm::Register lhs, int64_t imm)
  {
    emitMC(llvm::RISCV::SRAI, { mcReg(dst), mcReg(lhs), mcImm(imm) });
  }

  // 记录一次函数调用；真正把参数放到 a 寄存器里是在后面的 emit 阶段完成。
  void emitVCall(llvm::StringRef callee,
                 const llvm::SmallVectorImpl<llvm::Register>& args,
                 std::optional<llvm::Register> ret)
  {
    VInst vi;
    vi.kind = VInst::Kind::kCall;
    vi.sym = callee.str();
    vi.args.append(args.begin(), args.end());
    if (ret.has_value()) {
      vi.has_ret = true;
      vi.ret = *ret;
    }
    insts_.push_back(std::move(vi));
  }

  // 记录一次函数返回；返回值若存在，也先记成虚拟寄存器。
  void emitVRet(std::optional<llvm::Register> ret)
  {
    VInst vi;
    vi.kind = VInst::Kind::kRet;
    if (ret.has_value()) {
      vi.has_ret = true;
      vi.ret = *ret;
    }
    insts_.push_back(std::move(vi));
  }

  // 记录无条件跳转。
  void emitVBranchUncond(llvm::StringRef target)
  {
    VInst vi;
    vi.kind = VInst::Kind::kBrUncond;
    vi.sym = target.str();
    insts_.push_back(std::move(vi));
  }

  // 记录条件跳转，branch_on_zero 表示“条件为 0 时跳”还是“非 0 时跳”。
  void emitVBranchCond(llvm::Register cond, llvm::StringRef target, bool branch_on_zero)
  {
    VInst vi;
    vi.kind = VInst::Kind::kBrCond;
    vi.sym = target.str();
    vi.cond = cond;
    vi.branch_on_zero = branch_on_zero;
    insts_.push_back(std::move(vi));
  }

  // 把一个 LLVM 值变成“当前可用的寄存器值”。
  // 这是 emitInst 最常走的入口：常量要先装入临时寄存器，普通 SSA 值直接查表。
  llvm::Register emitLoadValue(const llvm::Value* v)
  {
    if (llvm::isa<llvm::UndefValue>(v) || llvm::isa<llvm::PoisonValue>(v) ||
        llvm::isa<llvm::ConstantPointerNull>(v) || llvm::isa<llvm::ConstantAggregateZero>(v)) {
      llvm::Register tmp = nextTempVReg();
      emitVLoadImm(tmp, 0);
      return tmp;
    }
    if (const auto* ci = llvm::dyn_cast<llvm::ConstantInt>(v)) {
      llvm::Register tmp = nextTempVReg();
      int64_t imm = ci->getBitWidth() == 1 ? static_cast<int64_t>(ci->getZExtValue())
                                           : ci->getSExtValue();
      emitVLoadImm(tmp, imm);
      return tmp;
    }
    if (const auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(v)) {
      return emitConstExpr(*ce);
    }
    if (const auto* gv = llvm::dyn_cast<llvm::GlobalValue>(v)) {
      llvm::Register tmp = nextTempVReg();
      emitVLoadAddr(tmp, gv->getName());
      return tmp;
    }
    return vregOf(v);
  }

  // 处理出现在操作数里的 ConstantExpr，把它拆成普通的地址/寄存器计算。
  llvm::Register emitConstExpr(const llvm::ConstantExpr& ce)
  {
    switch (ce.getOpcode()) {
      case llvm::Instruction::BitCast:
      case llvm::Instruction::PtrToInt:
      case llvm::Instruction::IntToPtr:
        return emitLoadValue(ce.getOperand(0));
      case llvm::Instruction::GetElementPtr: {
        llvm::Register dst = nextTempVReg();
        llvm::Register base = emitLoadValue(ce.getOperand(0));
        emitVMov(dst, base);

        int64_t constOff = 0;
        auto typeIt = llvm::gep_type_begin(ce);
        auto idxIt = ce.op_begin() + 1;
        for (; typeIt != llvm::gep_type_end(ce); ++typeIt, ++idxIt) {
          if (typeIt.isStruct())
            llvm::report_fatal_error("struct GEP in ConstantExpr not supported");
          const auto* idxC = llvm::dyn_cast<llvm::ConstantInt>(*idxIt);
          if (!idxC)
            llvm::report_fatal_error("non-constant index in ConstantExpr GEP");
          int64_t stride = static_cast<int64_t>(
              typeIt.getSequentialElementStride(dl_).getFixedValue());
          constOff += idxC->getSExtValue() * stride;
        }

        if (constOff != 0)
          emitVAddImm(dst, dst, constOff);
        return dst;
      }
      default:
        llvm::report_fatal_error(llvm::Twine("unsupported ConstantExpr opcode: ") +
                                 ce.getOpcodeName());
    }
    return llvm::Register();
  }

  // 按指令种类分发到具体的 lowering 逻辑。
  void emitInst(const llvm::Instruction& inst)
  {
    if (const auto* ai = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
      int32_t objOff = allocaObjectOff(*ai);
      llvm::Register dst = vregOf(ai);
      emitVAddImmSp(dst, objOff);
      return;
    }

    if (const auto* bo = llvm::dyn_cast<llvm::BinaryOperator>(&inst)) {
      emitBinary(*bo);
      return;
    }

    if (const auto* li = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
      emitLoadInst(*li);
      return;
    }

    if (const auto* si = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
      emitStoreInst(*si);
      return;
    }

    if (const auto* ci = llvm::dyn_cast<llvm::CallInst>(&inst)) {
      emitCallInst(*ci);
      return;
    }

    if (const auto* ri = llvm::dyn_cast<llvm::ReturnInst>(&inst)) {
      emitReturnInst(*ri);
      return;
    }

    if (const auto* bi = llvm::dyn_cast<llvm::BranchInst>(&inst)) {
      emitBranchInst(*bi);
      return;
    }

    if (const auto* si = llvm::dyn_cast<llvm::SwitchInst>(&inst)) {
      emitSwitchInst(*si);
      return;
    }

    if (const auto* ci = llvm::dyn_cast<llvm::ICmpInst>(&inst)) {
      emitICmpInst(*ci);
      return;
    }

    if (const auto* si = llvm::dyn_cast<llvm::SelectInst>(&inst)) {
      emitSelectInst(*si);
      return;
    }

    if (const auto* gi = llvm::dyn_cast<llvm::GetElementPtrInst>(&inst)) {
      emitGEPInst(*gi);
      return;
    }

    if (llvm::isa<llvm::BitCastInst>(inst) || llvm::isa<llvm::ZExtInst>(inst) ||
        llvm::isa<llvm::SExtInst>(inst) || llvm::isa<llvm::TruncInst>(inst) ||
        llvm::isa<llvm::PtrToIntInst>(inst) || llvm::isa<llvm::IntToPtrInst>(inst)) {
      llvm::Register src = emitLoadValue(inst.getOperand(0));
      llvm::Register dst = vregOf(&inst);
      emitVMov(dst, src);
      return;
    }

    llvm::report_fatal_error(llvm::Twine("unsupported instruction: ") + inst.getOpcodeName());
  }

  /*
  ----------------------------------------------------------------
  TASK 5 START
  ----------------------------------------------------------------
  */

  // 翻译二元运算：先取出左右操作数，再挑选对应的 RV64 指令。
  //
  // 建议按下面节奏实现：
  // 1) 先把左值、右值、目标位置统一准备好，后续所有分支都复用这三者。
  // 2) 以 opcode 做分发，优先把“加减乘除余、与或异或”这几类一对一语义先打通。
  // 3) 三种移位要单独处理：右操作数如果是编译期常量，走立即数字段更自然；
  //    如果不是常量，就按“寄存器给位数”的路径实现。
  // 4) 注意区分逻辑右移和算术右移，二者在负数输入上的行为不同。
  // 5) 未覆盖到的 opcode 明确报错，不要静默吞掉，这样测例更容易定位问题。
  // 6) 建议先保证语义正确，再考虑可读性和精简，不要在这里提前做优化。
  // 7) 自检时至少覆盖：负数除法/取余、较大移位位数、以及位运算混合场景。
  void emitBinary(const llvm::BinaryOperator& bo)
  {
    llvm::Register lhs = emitLoadValue(bo.getOperand(0));
    llvm::Register rhs = emitLoadValue(bo.getOperand(1));
    llvm::Register dst = vregOf(&bo);

    auto emitShift = [&](auto emitReg, auto emitImm) {
      if (const auto* ci = llvm::dyn_cast<llvm::ConstantInt>(bo.getOperand(1))) {
        (this->*emitImm)(dst, lhs, ci->getSExtValue());
      } else {
        (this->*emitReg)(dst, lhs, rhs);
      }
    };

    switch (bo.getOpcode()) {
      case llvm::Instruction::Add:
        emitVAdd(dst, lhs, rhs);
        return;
      case llvm::Instruction::Sub:
        emitVSub(dst, lhs, rhs);
        return;
      case llvm::Instruction::Mul:
        emitVMul(dst, lhs, rhs);
        return;
      case llvm::Instruction::SDiv:
        emitVDiv(dst, lhs, rhs);
        return;
      case llvm::Instruction::SRem:
        emitVRem(dst, lhs, rhs);
        return;
      case llvm::Instruction::And:
        emitVAnd(dst, lhs, rhs);
        return;
      case llvm::Instruction::Or:
        emitVOr(dst, lhs, rhs);
        return;
      case llvm::Instruction::Xor:
        emitVXor(dst, lhs, rhs);
        return;
      case llvm::Instruction::Shl:
        emitShift(&FuncLowering::emitVSll, &FuncLowering::emitVSlli);
        return;
      case llvm::Instruction::LShr:
        emitShift(&FuncLowering::emitVSrl, &FuncLowering::emitVSrli);
        return;
      case llvm::Instruction::AShr:
        emitShift(&FuncLowering::emitVSra, &FuncLowering::emitVSrai);
        return;
      default:
        llvm::report_fatal_error(llvm::Twine("unsupported binary opcode: ") +
                                 bo.getOpcodeName());
    }
  }

  // 翻译整数比较，目标是把结果变成 0/1，方便后续 branch/select 继续使用。
  //
  // 建议按“先归一化，再分类型”的思路实现：
  // 1) 先把左右输入和结果位置准备好，再留一个临时寄存器做中间值。
  // 2) 先统一目标：所有比较最终都写成 0/1，避免后续分支和 select 出现歧义。
  // 3) 相等/不等可以先构造“差异是否为 0”的中间结论，再转成布尔值。
  // 4) 严格小于类比较可直接使用底层“小于”语义；“大于”可通过交换左右得到。
  // 5) 小于等于/大于等于建议用“严格比较后再取反”的方式，逻辑更稳定。
  // 6) 有符号和无符号要分两套路径，尤其要覆盖最高位为 1 的输入。
  // 7) 每个谓词分支都应明确结束；未支持谓词直接报错，避免产生错误代码。
  // 8) 自检重点：-1 与 0、INT_MIN 边界、以及无符号大数比较。
  void emitICmpInst(const llvm::ICmpInst& ci)
  {
    llvm::Register lhs = emitLoadValue(ci.getOperand(0));
    llvm::Register rhs = emitLoadValue(ci.getOperand(1));
    llvm::Register dst = vregOf(&ci);
    llvm::Register tmp = nextTempVReg();

    switch (ci.getPredicate()) {
      case llvm::CmpInst::ICMP_EQ:
        emitVXor(tmp, lhs, rhs);
        emitVSltiu(dst, tmp, 1);
        return;
      case llvm::CmpInst::ICMP_NE:
        emitVXor(tmp, lhs, rhs);
        emitVSltu(dst, llvm::Register(llvm::RISCV::X0), tmp);
        return;
      case llvm::CmpInst::ICMP_SLT:
        emitVSlt(dst, lhs, rhs);
        return;
      case llvm::CmpInst::ICMP_SGT:
        emitVSlt(dst, rhs, lhs);
        return;
      case llvm::CmpInst::ICMP_SLE:
        emitVSlt(tmp, rhs, lhs);
        emitVXori(dst, tmp, 1);
        return;
      case llvm::CmpInst::ICMP_SGE:
        emitVSlt(tmp, lhs, rhs);
        emitVXori(dst, tmp, 1);
        return;
      case llvm::CmpInst::ICMP_ULT:
        emitVSltu(dst, lhs, rhs);
        return;
      case llvm::CmpInst::ICMP_UGT:
        emitVSltu(dst, rhs, lhs);
        return;
      case llvm::CmpInst::ICMP_ULE:
        emitVSltu(tmp, rhs, lhs);
        emitVXori(dst, tmp, 1);
        return;
      case llvm::CmpInst::ICMP_UGE:
        emitVSltu(tmp, lhs, rhs);
        emitVXori(dst, tmp, 1);
        return;
      default:
        llvm::report_fatal_error("unsupported icmp predicate");
    }
  }

  // 翻译 load：先算地址，再按类型大小选择 ld 或 lw。
  //
  // 推荐实现步骤：
  // 1) 先把“从哪里读”与“读到哪里”确定下来，再处理位宽选择。
  // 2) 根据被加载值的类型大小决定读宽度：当前实验只需区分 4 字节与非 4 字节。
  // 3) 4 字节路径走 32 位读取，其余按 64 位读取处理。
  // 4) 地址本身默认已在前序阶段算好，这里只做最终发射，不做额外折叠优化。
  // 5) 自检时关注 i32 与 i64 混合读取，确认符号位行为符合预期。
  void emitLoadInst(const llvm::LoadInst& li)
  {
    llvm::Register addr = emitLoadValue(li.getPointerOperand());
    llvm::Register dst = vregOf(&li);
    uint64_t size = dl_.getTypeAllocSize(li.getType());

    if (size == 4) {
      emitVLoad32(dst, addr);
    } else {
      emitVLoad(dst, addr);
    }
  }

  // 翻译 store：先准备值和地址，再按大小选择 sd 或 sw。
  //
  // 推荐实现步骤：
  // 1) 先分别准备“要写入的数据”和“写入目标地址”。
  // 2) 根据写入值的类型大小选择写宽度：4 字节走 32 位存储，其余走 64 位存储。
  // 3) 保持与 load 的宽度规则对称，避免读写不匹配导致的隐性错误。
  // 4) 该函数聚焦在指令选择，不负责额外的地址安全检查与别名分析。
  // 5) 自检时建议覆盖：连续写入同一地址、i32 覆盖写、以及全局变量写入。
  void emitStoreInst(const llvm::StoreInst& si)
  {
    llvm::Register value = emitLoadValue(si.getValueOperand());
    llvm::Register addr = emitLoadValue(si.getPointerOperand());
    uint64_t size = dl_.getTypeAllocSize(si.getValueOperand()->getType());

    if (size == 4) {
      emitVStore32(value, addr);
    } else {
      emitVStore(value, addr);
    }
  }

   /*
  ----------------------------------------------------------------
  TASK 5 END
  ----------------------------------------------------------------
  */

  // 翻译函数调用；这里先收集参数和返回值，具体调用约定留给后面的 emit 阶段处理。
  void emitCallInst(const llvm::CallInst& ci)
  {
    const llvm::Value* called = ci.getCalledOperand()->stripPointerCasts();
    const llvm::Function* callee = llvm::dyn_cast<llvm::Function>(called);
    if (!callee)
      llvm::report_fatal_error("indirect call not supported");
    if (callee->isIntrinsic()) {
      switch (callee->getIntrinsicID()) {
        case llvm::Intrinsic::lifetime_start:
        case llvm::Intrinsic::lifetime_end:
          return;
        case llvm::Intrinsic::smax:
        case llvm::Intrinsic::smin: {
          if (ci.arg_size() != 2)
            llvm::report_fatal_error("unexpected smin/smax arg count");
          llvm::Register lhs = emitLoadValue(ci.getArgOperand(0));
          llvm::Register rhs = emitLoadValue(ci.getArgOperand(1));
          llvm::Register dst = vregOf(&ci);
          llvm::Register cond = nextTempVReg();
          emitVSlt(cond, lhs, rhs);
          if (callee->getIntrinsicID() == llvm::Intrinsic::smax) {
            // dst = lhs < rhs ? rhs : lhs
            llvm::Register tmp = nextTempVReg();
            llvm::Register tmp2 = nextTempVReg();
            emitVSub(tmp, rhs, lhs);
            emitVMul(tmp2, cond, tmp);
            emitVAdd(dst, lhs, tmp2);
          } else {
            // dst = lhs < rhs ? lhs : rhs
            llvm::Register tmp = nextTempVReg();
            llvm::Register tmp2 = nextTempVReg();
            emitVSub(tmp, lhs, rhs);
            emitVMul(tmp2, cond, tmp);
            emitVAdd(dst, rhs, tmp2);
          }
          return;
        }
        default:
          llvm::report_fatal_error("unsupported intrinsic in rv64-min backend");
      }
    }

    llvm::SmallVector<llvm::Register, 8> args;
    args.reserve(ci.arg_size());
    for (unsigned i = 0; i < ci.arg_size(); ++i) {
      args.push_back(emitLoadValue(ci.getArgOperand(i)));
    }

    std::optional<llvm::Register> ret;
    if (!ci.getType()->isVoidTy())
      ret = vregOf(&ci);

    emitVCall(callee->getName(), args, ret);
  }

  // 翻译 return，若有返回值就先把它准备好。
  void emitReturnInst(const llvm::ReturnInst& ri)
  {
    if (const llvm::Value* rv = ri.getReturnValue()) {
      emitVRet(emitLoadValue(rv));
      return;
    }
    emitVRet(std::nullopt);
  }

  // 翻译 select，这里把“条件选择”改写成算术组合，避免额外控制流。
  void emitSelectInst(const llvm::SelectInst& si)
  {
    llvm::Register cond = emitLoadValue(si.getCondition());
    llvm::Register tv = emitLoadValue(si.getTrueValue());
    llvm::Register fv = emitLoadValue(si.getFalseValue());
    llvm::Register dst = vregOf(&si);

    llvm::Register tmp = nextTempVReg();
    llvm::Register tmp2 = nextTempVReg();
    emitVSub(tmp, tv, fv);
    emitVMul(tmp2, cond, tmp);
    emitVAdd(dst, fv, tmp2);
  }

  // 翻译 br，同时处理 PHI 需要的边上复制。
  // 关键点是：真正的 PHI move 不放在前驱块末尾，而是放进单独的 edge block。
  void emitBranchInst(const llvm::BranchInst& bi)
  {
    if (!curBB_)
      llvm::report_fatal_error("missing current basic block");

    if (bi.isUnconditional()) {
      const llvm::BasicBlock* succ = bi.getSuccessor(0);
      auto moves = collectPhiMoves(*curBB_, *succ);
      std::string target = bbLabel(*succ);
      if (!moves.empty()) {
        std::string edgeLabel = makeEdgeLabel();
        pendingEdges_.push_back(EdgeBlock{edgeLabel, target, std::move(moves)});
        emitVBranchUncond(edgeLabel);
      } else {
        emitVBranchUncond(target);
      }
      return;
    }

    llvm::Register cond = emitLoadValue(bi.getCondition());
    const llvm::BasicBlock* tbb = bi.getSuccessor(0);
    const llvm::BasicBlock* fbb = bi.getSuccessor(1);

    auto tMoves = collectPhiMoves(*curBB_, *tbb);
    auto fMoves = collectPhiMoves(*curBB_, *fbb);

    std::string tLabel = tMoves.empty() ? bbLabel(*tbb) : makeEdgeLabel();
    std::string fLabel = fMoves.empty() ? bbLabel(*fbb) : makeEdgeLabel();

    if (!tMoves.empty())
      pendingEdges_.push_back(EdgeBlock{tLabel, bbLabel(*tbb), std::move(tMoves)});
    if (!fMoves.empty())
      pendingEdges_.push_back(EdgeBlock{fLabel, bbLabel(*fbb), std::move(fMoves)});

    emitVBranchCond(cond, tLabel, false);
    emitVBranchUncond(fLabel);
  }

  // 翻译 switch：逐个 case 比较，相等就跳走，最后落到 default。
  // 它和条件分支一样，也要照顾目标块里的 PHI 赋值。
  void emitSwitchInst(const llvm::SwitchInst& si)
  {
    if (!curBB_)
      llvm::report_fatal_error("missing current basic block");

    llvm::Register val = emitLoadValue(si.getCondition());
    for (const auto& c : si.cases()) {
      int64_t imm = c.getCaseValue()->getSExtValue();
      llvm::Register caseReg = nextTempVReg();
      emitVLoadImm(caseReg, imm);
      llvm::Register tmp = nextTempVReg();
      emitVXor(tmp, val, caseReg);
      llvm::Register iseq = nextTempVReg();
      emitVSltiu(iseq, tmp, 1);

      const llvm::BasicBlock* succ = c.getCaseSuccessor();
      auto moves = collectPhiMoves(*curBB_, *succ);
      std::string label = moves.empty() ? bbLabel(*succ) : makeEdgeLabel();
      if (!moves.empty())
        pendingEdges_.push_back(EdgeBlock{label, bbLabel(*succ), std::move(moves)});
      emitVBranchCond(iseq, label, false);
    }

    const llvm::BasicBlock* def = si.getDefaultDest();
    auto defMoves = collectPhiMoves(*curBB_, *def);
    std::string defLabel = defMoves.empty() ? bbLabel(*def) : makeEdgeLabel();
    if (!defMoves.empty())
      pendingEdges_.push_back(EdgeBlock{defLabel, bbLabel(*def), std::move(defMoves)});
    emitVBranchUncond(defLabel);
  }

  // 收集从 pred 进入 succ 时需要执行的 PHI move。
  // 这一步只“记下来”，真正发指令在 emitPhiMoves 里完成。
  llvm::SmallVector<PhiMove, 4> collectPhiMoves(const llvm::BasicBlock& pred,
                                                const llvm::BasicBlock& succ) const
  {
    llvm::SmallVector<PhiMove, 4> moves;
    for (const llvm::Instruction& inst : succ) {
      const auto* phi = llvm::dyn_cast<llvm::PHINode>(&inst);
      if (!phi)
        break;
      const llvm::Value* incoming = phi->getIncomingValueForBlock(&pred);
      if (!incoming)
        llvm::report_fatal_error("missing phi incoming value");
      moves.push_back(PhiMove{vregOf(phi), incoming});
    }
    return moves;
  }

  // 把一组 PHI move 展开成普通寄存器复制。
  void emitPhiMoves(const llvm::SmallVectorImpl<PhiMove>& moves)
  {
    for (const PhiMove& mv : moves) {
      llvm::Register src = emitLoadValue(mv.src);
      if (src == mv.dst)
        continue;
      emitVMov(mv.dst, src);
    }
  }

  // 给边块生成一个独立标签，供 branch/switch 插入中转块时使用。
  std::string makeEdgeLabel()
  {
    return ".Ledge." + std::to_string(edgeId_++);
  }

  // 翻译 GEP：从基址出发，按元素步长不断累加偏移，得到最终地址。
  void emitGEPInst(const llvm::GetElementPtrInst& gi)
  {
    llvm::Register dst = vregOf(&gi);
    llvm::Register base = emitLoadValue(gi.getPointerOperand());
    emitVMov(dst, base);

    int64_t constOff = 0;
    auto typeIt = llvm::gep_type_begin(gi);
    auto idxIt = gi.idx_begin();
    for (; typeIt != llvm::gep_type_end(gi); ++typeIt, ++idxIt) {
      if (typeIt.isStruct())
        llvm::report_fatal_error("struct GEP not supported in rv64-min backend");
      int64_t stride = static_cast<int64_t>(
          typeIt.getSequentialElementStride(dl_).getFixedValue());
      const auto* idxC = llvm::dyn_cast<llvm::ConstantInt>(*idxIt);
      if (idxC) {
        constOff += idxC->getSExtValue() * stride;
        continue;
      }
      if (constOff != 0) {
        emitVAddImm(dst, dst, constOff);
        constOff = 0;
      }
      llvm::Register idxReg = emitLoadValue(*idxIt);
      llvm::Register scale = nextTempVReg();
      emitVLoadImm(scale, stride);
      emitVMul(scale, idxReg, scale);
      emitVAdd(dst, dst, scale);
    }

    if (constOff != 0)
      emitVAddImm(dst, dst, constOff);
  }
};

// 输出全局变量的数据段/零初始化段。
static void emitGlobals(const llvm::Module& m, RvAsmEmitter& as)
{
  const llvm::DataLayout& dl = m.getDataLayout();

  for (const llvm::GlobalVariable& gv : m.globals()) {
    if (gv.isDeclaration())
      continue;

    std::string name = gv.getName().str();
    if (name.empty())
      continue;

    uint64_t size = dl.getTypeAllocSize(gv.getValueType());

    if (gv.hasInitializer() && gv.getInitializer()->isZeroValue()) {
      as.emitDirective("\t.bss");
      if (gv.hasExternalLinkage())
        as.emitDirective("\t.globl " + name);
      as.emitDirective("\t.align 3");
      as.emitLabel(name);
      as.emitDirective("\t.zero " + std::to_string(size));
      continue;
    }

    as.emitDirective("\t.data");
    if (gv.hasExternalLinkage())
      as.emitDirective("\t.globl " + name);
    as.emitDirective("\t.align 3");
    as.emitLabel(name);

    const llvm::Constant* init = gv.hasInitializer() ? gv.getInitializer() : nullptr;
    if (!init) {
      as.emitDirective("\t.zero " + std::to_string(size));
      continue;
    }

    auto emitInit = [&](const llvm::Constant* c, auto&& emitInitRef) -> void {
      if (c->isZeroValue()) {
        as.emitDirective("\t.zero " + std::to_string(dl.getTypeAllocSize(c->getType())));
        return;
      }
      if (const auto* ci = llvm::dyn_cast<llvm::ConstantInt>(c)) {
        uint64_t sz = dl.getTypeAllocSize(ci->getType());
        if (sz == 4) {
          as.emitDirective("\t.word " + std::to_string(ci->getSExtValue()));
        } else if (sz == 8) {
          as.emitDirective("\t.quad " + std::to_string(ci->getSExtValue()));
        } else if (sz == 1) {
          as.emitDirective("\t.byte " + std::to_string(ci->getZExtValue()));
        } else {
          as.emitDirective("\t.zero " + std::to_string(sz));
        }
        return;
      }
      if (const auto* ca = llvm::dyn_cast<llvm::ConstantArray>(c)) {
        for (unsigned i = 0; i < ca->getNumOperands(); ++i) {
          emitInitRef(llvm::cast<llvm::Constant>(ca->getOperand(i)), emitInitRef);
        }
        return;
      }
      if (const auto* cda = llvm::dyn_cast<llvm::ConstantDataArray>(c)) {
        for (unsigned i = 0; i < cda->getNumElements(); ++i) {
          emitInitRef(cda->getElementAsConstant(i), emitInitRef);
        }
        return;
      }
      as.emitDirective("\t.zero " + std::to_string(dl.getTypeAllocSize(c->getType())));
    };

    emitInit(init, emitInit);
  }
}

} // namespace

// 模块级入口：先输出全局变量，再逐个输出函数。
void emitModule(const llvm::Module& mod, llvm::raw_ostream& out)
{
  RvAsmEmitter as(out);
  emitGlobals(mod, as);

  FuncLowering lower(mod, as);
  for (const llvm::Function& f : mod.functions())
    lower.emitFunction(f);
}
