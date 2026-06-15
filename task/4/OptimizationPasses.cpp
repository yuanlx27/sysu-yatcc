#include "OptimizationPasses.hpp"

#include <map>
#include <tuple>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace {

bool
replaceInstruction(Instruction& inst, Value* replacement)
{
  if (!inst.use_empty())
    inst.replaceAllUsesWith(replacement);
  inst.eraseFromParent();
  return true;
}

ConstantInt*
asConstantInt(Value* value)
{
  return dyn_cast<ConstantInt>(value);
}

bool
isZero(Value* value)
{
  auto* constant = asConstantInt(value);
  return constant && constant->isZero();
}

bool
isOne(Value* value)
{
  auto* constant = asConstantInt(value);
  return constant && constant->isOne();
}

bool
isAllOnes(Value* value)
{
  auto* constant = asConstantInt(value);
  return constant && constant->isMinusOne();
}

PreservedAnalyses
preserved(bool changed)
{
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool
propagateGlobal(GlobalVariable& global)
{
  if (!global.hasInitializer() || global.isExternallyInitialized())
    return false;

  Constant* initializer = global.getInitializer();
  if (!initializer->getType()->isFirstClassType())
    return false;

  SmallVector<LoadInst*, 16> loads;
  for (User* user : global.users()) {
    auto* load = dyn_cast<LoadInst>(user);
    if (!load || load->getPointerOperand() != &global || load->isVolatile())
      return false;
    loads.push_back(load);
  }

  for (LoadInst* load : loads)
    replaceInstruction(*load, initializer);
  return !loads.empty();
}

bool
simplifyPhi(PHINode& phi)
{
  Value* common = nullptr;
  for (Value* incoming : phi.incoming_values()) {
    if (incoming == &phi)
      continue;
    if (!common)
      common = incoming;
    else if (incoming != common)
      return false;
  }
  return common && replaceInstruction(phi, common);
}

bool
simplifyInstruction(Instruction& inst)
{
  if (auto* phi = dyn_cast<PHINode>(&inst))
    return simplifyPhi(*phi);

  auto* binary = dyn_cast<BinaryOperator>(&inst);
  if (!binary)
    return false;

  Value* lhs = binary->getOperand(0);
  Value* rhs = binary->getOperand(1);
  Value* replacement = nullptr;

  switch (binary->getOpcode()) {
    case Instruction::Add:
      if (isZero(lhs))
        replacement = rhs;
      else if (isZero(rhs))
        replacement = lhs;
      break;
    case Instruction::Sub:
      if (isZero(rhs))
        replacement = lhs;
      else if (lhs == rhs)
        replacement = ConstantInt::get(binary->getType(), 0);
      break;
    case Instruction::Mul:
      if (isZero(lhs) || isZero(rhs))
        replacement = ConstantInt::get(binary->getType(), 0);
      else if (isOne(lhs))
        replacement = rhs;
      else if (isOne(rhs))
        replacement = lhs;
      break;
    case Instruction::UDiv:
    case Instruction::SDiv:
      if (isZero(lhs))
        replacement = ConstantInt::get(binary->getType(), 0);
      else if (isOne(rhs))
        replacement = lhs;
      break;
    case Instruction::URem:
    case Instruction::SRem:
      if (isZero(lhs) || isOne(rhs))
        replacement = ConstantInt::get(binary->getType(), 0);
      break;
    case Instruction::And:
      if (isZero(lhs) || isZero(rhs))
        replacement = ConstantInt::get(binary->getType(), 0);
      else if (isAllOnes(lhs))
        replacement = rhs;
      else if (isAllOnes(rhs) || lhs == rhs)
        replacement = lhs;
      break;
    case Instruction::Or:
      if (isZero(lhs))
        replacement = rhs;
      else if (isZero(rhs) || lhs == rhs)
        replacement = lhs;
      break;
    case Instruction::Xor:
      if (isZero(lhs))
        replacement = rhs;
      else if (isZero(rhs))
        replacement = lhs;
      else if (lhs == rhs)
        replacement = ConstantInt::get(binary->getType(), 0);
      break;
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
      if (isZero(rhs))
        replacement = lhs;
      break;
    default:
      break;
  }

  return replacement && replaceInstruction(inst, replacement);
}

bool
reduceStrength(BinaryOperator& binary)
{
  if (!binary.getType()->isIntegerTy())
    return false;

  Value* value = binary.getOperand(0);
  auto* divisor = asConstantInt(binary.getOperand(1));

  if (binary.getOpcode() == Instruction::Mul) {
    auto* lhsConstant = asConstantInt(value);
    if (lhsConstant) {
      divisor = lhsConstant;
      value = binary.getOperand(1);
    }
  }

  if (!divisor || divisor->isZero())
    return false;

  const APInt& amount = divisor->getValue();
  unsigned bitWidth = amount.getBitWidth();

  if (binary.getOpcode() == Instruction::Mul && amount.isPowerOf2()) {
    auto* shiftAmount = ConstantInt::get(binary.getType(), amount.logBase2());
    auto* shift =
      BinaryOperator::CreateShl(value, shiftAmount, binary.getName(), &binary);
    shift->setHasNoUnsignedWrap(binary.hasNoUnsignedWrap());
    shift->setHasNoSignedWrap(binary.hasNoSignedWrap());
    return replaceInstruction(binary, shift);
  }

  if (binary.getOpcode() == Instruction::UDiv && amount.isPowerOf2()) {
    if (binary.isExact())
      return false;
    auto* shiftAmount = ConstantInt::get(binary.getType(), amount.logBase2());
    auto* shift =
      BinaryOperator::CreateLShr(value, shiftAmount, binary.getName(), &binary);
    return replaceInstruction(binary, shift);
  }

  if (binary.getOpcode() == Instruction::URem && amount.isPowerOf2()) {
    auto* mask = ConstantInt::get(binary.getType(), amount - 1);
    auto* andInst =
      BinaryOperator::CreateAnd(value, mask, binary.getName(), &binary);
    return replaceInstruction(binary, andInst);
  }

  if (binary.getOpcode() == Instruction::SDiv && amount.isPowerOf2() &&
      !amount.isNegative() && !binary.isExact()) {
    auto* signShift = ConstantInt::get(binary.getType(), bitWidth - 1);
    auto* sign = BinaryOperator::CreateAShr(value, signShift, "", &binary);
    auto* mask = ConstantInt::get(binary.getType(), amount - 1);
    auto* bias = BinaryOperator::CreateAnd(sign, mask, "", &binary);
    auto* adjusted = BinaryOperator::CreateAdd(value, bias, "", &binary);
    auto* quotient = BinaryOperator::CreateAShr(
      adjusted,
      ConstantInt::get(binary.getType(), amount.logBase2()),
      binary.getName(),
      &binary);
    return replaceInstruction(binary, quotient);
  }

  return false;
}

bool
combineConstants(BinaryOperator& binary)
{
  auto* outerConstant = asConstantInt(binary.getOperand(1));
  auto* inner = dyn_cast<BinaryOperator>(binary.getOperand(0));
  if (!outerConstant || !inner || !inner->hasOneUse())
    return false;

  auto* innerConstant = asConstantInt(inner->getOperand(1));
  if (!innerConstant || inner->getOpcode() != binary.getOpcode())
    return false;

  unsigned opcode = binary.getOpcode();
  if (opcode != Instruction::Add && opcode != Instruction::Mul &&
      opcode != Instruction::And && opcode != Instruction::Or &&
      opcode != Instruction::Xor)
    return false;

  const APInt& lhs = innerConstant->getValue();
  const APInt& rhs = outerConstant->getValue();
  APInt combined = lhs;
  switch (opcode) {
    case Instruction::Add: {
      bool overflow = false;
      combined = lhs.sadd_ov(rhs, overflow);
      if (overflow || (lhs.isNegative() != rhs.isNegative()))
        return false;
      break;
    }
    case Instruction::Mul: {
      bool signedOverflow = false;
      bool unsignedOverflow = false;
      APInt signedProduct = lhs.smul_ov(rhs, signedOverflow);
      APInt unsignedProduct = lhs.umul_ov(rhs, unsignedOverflow);
      if (((binary.hasNoSignedWrap() || inner->hasNoSignedWrap()) &&
           signedOverflow) ||
          ((binary.hasNoUnsignedWrap() || inner->hasNoUnsignedWrap()) &&
           unsignedOverflow))
        return false;
      combined = (binary.hasNoUnsignedWrap() || inner->hasNoUnsignedWrap())
                   ? unsignedProduct
                   : signedProduct;
      break;
    }
    case Instruction::And:
      combined &= rhs;
      break;
    case Instruction::Or:
      combined |= rhs;
      break;
    case Instruction::Xor:
      combined ^= rhs;
      break;
    default:
      llvm_unreachable("unexpected associative opcode");
  }

  auto* result =
    BinaryOperator::Create(static_cast<Instruction::BinaryOps>(opcode),
                           inner->getOperand(0),
                           ConstantInt::get(binary.getType(), combined),
                           binary.getName(),
                           &binary);

  if (opcode == Instruction::Add || opcode == Instruction::Mul) {
    result->setHasNoUnsignedWrap(binary.hasNoUnsignedWrap() &&
                                 inner->hasNoUnsignedWrap());
    result->setHasNoSignedWrap(binary.hasNoSignedWrap() &&
                               inner->hasNoSignedWrap());
  }

  replaceInstruction(binary, result);
  if (inner->use_empty())
    inner->eraseFromParent();
  return true;
}

struct ExpressionKey
{
  unsigned Opcode = 0;
  unsigned Flags = 0;
  Type* ResultType = nullptr;
  const void* Auxiliary = nullptr;
  SmallVector<Value*, 4> Operands;

  bool operator<(const ExpressionKey& other) const
  {
    auto head = std::tie(Opcode, Flags, ResultType, Auxiliary);
    auto otherHead =
      std::tie(other.Opcode, other.Flags, other.ResultType, other.Auxiliary);
    if (head != otherHead)
      return head < otherHead;

    if (Operands.size() != other.Operands.size())
      return Operands.size() < other.Operands.size();
    std::less<Value*> less;
    for (size_t i = 0; i < Operands.size(); ++i) {
      if (Operands[i] == other.Operands[i])
        continue;
      return less(Operands[i], other.Operands[i]);
    }
    return false;
  }
};

bool
makeExpressionKey(Instruction& inst, ExpressionKey& key)
{
  key.Opcode = inst.getOpcode();
  key.ResultType = inst.getType();

  if (auto* binary = dyn_cast<BinaryOperator>(&inst)) {
    if (!binary->getType()->isIntegerTy())
      return false;
    key.Operands.assign(binary->op_begin(), binary->op_end());
    if (binary->isCommutative() &&
        std::less<Value*>()(key.Operands[1], key.Operands[0]))
      std::swap(key.Operands[0], key.Operands[1]);
    if (auto* overflowing = dyn_cast<OverflowingBinaryOperator>(binary))
      key.Flags = unsigned(overflowing->hasNoUnsignedWrap()) |
                  (unsigned(overflowing->hasNoSignedWrap()) << 1);
    if (auto* exact = dyn_cast<PossiblyExactOperator>(binary))
      key.Flags |= unsigned(exact->isExact()) << 2;
    return true;
  }

  if (auto* cmp = dyn_cast<ICmpInst>(&inst)) {
    key.Flags = cmp->getPredicate();
    key.Operands.assign(cmp->op_begin(), cmp->op_end());
    if (cmp->isCommutative() &&
        std::less<Value*>()(key.Operands[1], key.Operands[0]))
      std::swap(key.Operands[0], key.Operands[1]);
    return true;
  }

  if (auto* cast = dyn_cast<CastInst>(&inst)) {
    key.Operands.push_back(cast->getOperand(0));
    return true;
  }

  if (auto* gep = dyn_cast<GetElementPtrInst>(&inst)) {
    key.Flags = gep->isInBounds();
    key.Auxiliary = gep->getSourceElementType();
    key.Operands.assign(gep->op_begin(), gep->op_end());
    return true;
  }

  return false;
}

bool
eliminateCommonSubexpressions(Function& function)
{
  DominatorTree dominators(function);
  std::map<ExpressionKey, SmallVector<Instruction*, 4>> leaders;
  SmallVector<Instruction*, 32> redundant;

  for (BasicBlock& block : function) {
    for (Instruction& inst : block) {
      ExpressionKey key;
      if (!makeExpressionKey(inst, key))
        continue;

      auto& candidates = leaders[key];
      Instruction* leader = nullptr;
      for (Instruction* candidate : candidates) {
        if (dominators.dominates(candidate, &inst)) {
          leader = candidate;
          break;
        }
      }

      if (!leader) {
        candidates.push_back(&inst);
        continue;
      }

      inst.replaceAllUsesWith(leader);
      redundant.push_back(&inst);
    }
  }

  for (Instruction* inst : redundant)
    inst->eraseFromParent();
  return !redundant.empty();
}

bool
hoistLoop(Loop& loop)
{
  bool changed = false;
  for (Loop* subLoop : loop.getSubLoops())
    changed |= hoistLoop(*subLoop);

  BasicBlock* preheader = loop.getLoopPreheader();
  if (!preheader)
    return changed;

  bool localChange;
  do {
    localChange = false;
    for (BasicBlock* block : loop.blocks()) {
      for (auto it = block->begin(), end = block->end(); it != end;) {
        Instruction& inst = *it++;
        if (isa<PHINode>(inst) || inst.isTerminator() ||
            inst.mayReadOrWriteMemory() || !isSafeToSpeculativelyExecute(&inst))
          continue;

        bool invariant = llvm::all_of(inst.operands(), [&loop](Use& operand) {
          return loop.isLoopInvariant(operand.get());
        });
        if (!invariant)
          continue;

        inst.moveBefore(preheader->getTerminator());
        localChange = true;
        changed = true;
      }
    }
  } while (localChange);
  return changed;
}

bool
eliminateDeadGlobals(Module& mod)
{
  bool changed = false;
  SmallVector<GlobalVariable*, 8> deadGlobals;

  for (GlobalVariable& global : mod.globals()) {
    SmallVector<StoreInst*, 16> stores;
    bool onlyStored = !global.isDeclaration();
    for (User* user : global.users()) {
      auto* store = dyn_cast<StoreInst>(user);
      if (!store || store->getPointerOperand() != &global ||
          store->isVolatile()) {
        onlyStored = false;
        break;
      }
      stores.push_back(store);
    }

    if (!onlyStored || stores.empty())
      continue;
    for (StoreInst* store : stores)
      store->eraseFromParent();
    if (global.use_empty() && global.hasLocalLinkage())
      deadGlobals.push_back(&global);
    changed |= !stores.empty();
  }

  for (GlobalVariable* global : deadGlobals)
    global->eraseFromParent();
  return changed || !deadGlobals.empty();
}

bool
eliminateDeadInstructions(Module& mod)
{
  bool changed = false;
  bool localChange;
  do {
    localChange = false;
    for (Function& function : mod) {
      for (BasicBlock& block : function) {
        for (auto it = block.rbegin(), end = block.rend(); it != end;) {
          Instruction& inst = *it++;
          if (!inst.use_empty() || inst.isTerminator() ||
              inst.mayHaveSideEffects())
            continue;
          inst.eraseFromParent();
          localChange = true;
          changed = true;
        }
      }
    }
  } while (localChange);
  return changed;
}

bool
simplifyBranches(Function& function)
{
  bool changed = false;
  for (BasicBlock& block : function) {
    auto* branch = dyn_cast<BranchInst>(block.getTerminator());
    if (!branch || !branch->isConditional())
      continue;
    auto* condition = asConstantInt(branch->getCondition());
    if (!condition)
      continue;

    BasicBlock* taken = branch->getSuccessor(condition->isZero() ? 1 : 0);
    BasicBlock* notTaken = branch->getSuccessor(condition->isZero() ? 0 : 1);
    notTaken->removePredecessor(&block);
    BranchInst::Create(taken, branch);
    branch->eraseFromParent();
    changed = true;
  }
  return changed;
}

bool
removeUnreachableBlocks(Function& function)
{
  SmallPtrSet<BasicBlock*, 32> reachable;
  SmallVector<BasicBlock*, 32> worklist;
  worklist.push_back(&function.getEntryBlock());

  while (!worklist.empty()) {
    BasicBlock* block = worklist.pop_back_val();
    if (!reachable.insert(block).second)
      continue;
    for (BasicBlock* successor : successors(block))
      worklist.push_back(successor);
  }

  SmallVector<BasicBlock*, 16> unreachable;
  for (BasicBlock& block : function)
    if (!reachable.contains(&block))
      unreachable.push_back(&block);

  for (BasicBlock* block : unreachable)
    for (BasicBlock* successor : successors(block))
      successor->removePredecessor(block);
  for (BasicBlock* block : unreachable)
    block->dropAllReferences();
  for (BasicBlock* block : unreachable)
    block->eraseFromParent();
  return !unreachable.empty();
}

bool
mergeLinearBlocks(Function& function)
{
  bool changed = false;
  bool localChange;
  do {
    localChange = false;
    for (auto blockIt = function.begin(), blockEnd = function.end();
         blockIt != blockEnd;
         ++blockIt) {
      BasicBlock& block = *blockIt;
      auto* branch = dyn_cast<BranchInst>(block.getTerminator());
      if (!branch || !branch->isUnconditional())
        continue;

      BasicBlock* successor = branch->getSuccessor(0);
      if (successor == &block || successor->hasAddressTaken() ||
          successor->getSinglePredecessor() != &block)
        continue;

      while (auto* phi = dyn_cast<PHINode>(&successor->front())) {
        Value* incoming = phi->getIncomingValueForBlock(&block);
        if (incoming == phi)
          incoming = UndefValue::get(phi->getType());
        replaceInstruction(*phi, incoming);
      }

      branch->eraseFromParent();
      successor->replaceAllUsesWith(&block);
      block.splice(block.end(), successor);
      successor->eraseFromParent();
      localChange = true;
      changed = true;
      break;
    }
  } while (localChange);
  return changed;
}

} // namespace

PreservedAnalyses
ConstantPropagation::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  for (GlobalVariable& global : mod.globals())
    changed |= propagateGlobal(global);
  return preserved(changed);
}

PreservedAnalyses
AlgebraicSimplification::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  bool localChange;
  do {
    localChange = false;
    for (Function& function : mod)
      for (BasicBlock& block : function)
        for (auto it = block.begin(), end = block.end(); it != end;) {
          Instruction& inst = *it++;
          localChange |= simplifyInstruction(inst);
        }
    changed |= localChange;
  } while (localChange);
  return preserved(changed);
}

PreservedAnalyses
StrengthReduction::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  for (Function& function : mod)
    for (BasicBlock& block : function)
      for (auto it = block.begin(), end = block.end(); it != end;) {
        Instruction& inst = *it++;
        if (auto* binary = dyn_cast<BinaryOperator>(&inst))
          changed |= reduceStrength(*binary);
      }
  return preserved(changed);
}

PreservedAnalyses
InstructionCombining::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  bool localChange;
  do {
    localChange = false;
    for (Function& function : mod)
      for (BasicBlock& block : function)
        for (auto it = block.begin(), end = block.end(); it != end;) {
          Instruction& inst = *it++;
          if (auto* binary = dyn_cast<BinaryOperator>(&inst))
            localChange |= combineConstants(*binary);
        }
    changed |= localChange;
  } while (localChange);
  return preserved(changed);
}

PreservedAnalyses
CommonSubexpressionElimination::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  for (Function& function : mod)
    if (!function.isDeclaration())
      changed |= eliminateCommonSubexpressions(function);
  return preserved(changed);
}

PreservedAnalyses
LoopInvariantCodeMotion::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  for (Function& function : mod) {
    if (function.isDeclaration())
      continue;
    DominatorTree dominators(function);
    LoopInfo loops(dominators);
    for (Loop* loop : loops)
      changed |= hoistLoop(*loop);
  }
  return preserved(changed);
}

PreservedAnalyses
DeadCodeElimination::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = eliminateDeadGlobals(mod);
  changed |= eliminateDeadInstructions(mod);
  return preserved(changed);
}

PreservedAnalyses
ControlFlowSimplification::run(Module& mod, ModuleAnalysisManager&)
{
  bool changed = false;
  for (Function& function : mod) {
    if (function.isDeclaration())
      continue;
    changed |= simplifyBranches(function);
    changed |= removeUnreachableBlocks(function);
    changed |= mergeLinearBlocks(function);
  }
  return preserved(changed);
}
