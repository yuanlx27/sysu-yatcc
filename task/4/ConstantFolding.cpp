#include "ConstantFolding.hpp"

#include <llvm/IR/Module.h>

using namespace llvm;

namespace {

Constant*
foldBinary(BinaryOperator& binary)
{
  auto* lhs = dyn_cast<ConstantInt>(binary.getOperand(0));
  auto* rhs = dyn_cast<ConstantInt>(binary.getOperand(1));
  if (!lhs || !rhs)
    return nullptr;

  const APInt& left = lhs->getValue();
  const APInt& right = rhs->getValue();
  APInt result = left;

  switch (binary.getOpcode()) {
    case Instruction::Add: {
      bool signedOverflow = false;
      bool unsignedOverflow = false;
      APInt signedResult = left.sadd_ov(right, signedOverflow);
      APInt unsignedResult = left.uadd_ov(right, unsignedOverflow);
      if ((binary.hasNoSignedWrap() && signedOverflow) ||
          (binary.hasNoUnsignedWrap() && unsignedOverflow))
        return PoisonValue::get(binary.getType());
      result = binary.hasNoUnsignedWrap() ? unsignedResult : signedResult;
      break;
    }
    case Instruction::Sub: {
      bool signedOverflow = false;
      bool unsignedOverflow = false;
      APInt signedResult = left.ssub_ov(right, signedOverflow);
      APInt unsignedResult = left.usub_ov(right, unsignedOverflow);
      if ((binary.hasNoSignedWrap() && signedOverflow) ||
          (binary.hasNoUnsignedWrap() && unsignedOverflow))
        return PoisonValue::get(binary.getType());
      result = binary.hasNoUnsignedWrap() ? unsignedResult : signedResult;
      break;
    }
    case Instruction::Mul: {
      bool signedOverflow = false;
      bool unsignedOverflow = false;
      APInt signedResult = left.smul_ov(right, signedOverflow);
      APInt unsignedResult = left.umul_ov(right, unsignedOverflow);
      if ((binary.hasNoSignedWrap() && signedOverflow) ||
          (binary.hasNoUnsignedWrap() && unsignedOverflow))
        return PoisonValue::get(binary.getType());
      result = binary.hasNoUnsignedWrap() ? unsignedResult : signedResult;
      break;
    }
    case Instruction::UDiv:
      if (right.isZero())
        return nullptr;
      if (binary.isExact() && !left.urem(right).isZero())
        return PoisonValue::get(binary.getType());
      result = left.udiv(right);
      break;
    case Instruction::SDiv:
      if (right.isZero() || (left.isMinSignedValue() && right.isAllOnes()))
        return nullptr;
      if (binary.isExact() && !left.srem(right).isZero())
        return PoisonValue::get(binary.getType());
      result = left.sdiv(right);
      break;
    case Instruction::URem:
      if (right.isZero())
        return nullptr;
      result = left.urem(right);
      break;
    case Instruction::SRem:
      if (right.isZero() || (left.isMinSignedValue() && right.isAllOnes()))
        return nullptr;
      result = left.srem(right);
      break;
    case Instruction::And:
      result &= right;
      break;
    case Instruction::Or:
      result |= right;
      break;
    case Instruction::Xor:
      result ^= right;
      break;
    case Instruction::Shl:
      if (right.uge(left.getBitWidth()))
        return nullptr;
      if (binary.hasNoSignedWrap() || binary.hasNoUnsignedWrap())
        return nullptr;
      result = left.shl(right.getLimitedValue());
      break;
    case Instruction::LShr:
      if (right.uge(left.getBitWidth()))
        return nullptr;
      if (binary.isExact())
        return nullptr;
      result = left.lshr(right.getLimitedValue());
      break;
    case Instruction::AShr:
      if (right.uge(left.getBitWidth()))
        return nullptr;
      if (binary.isExact())
        return nullptr;
      result = left.ashr(right.getLimitedValue());
      break;
    default:
      return nullptr;
  }
  return ConstantInt::get(binary.getType(), result);
}

Constant*
foldCompare(ICmpInst& compare)
{
  auto* lhs = dyn_cast<ConstantInt>(compare.getOperand(0));
  auto* rhs = dyn_cast<ConstantInt>(compare.getOperand(1));
  if (!lhs || !rhs)
    return nullptr;

  const APInt& left = lhs->getValue();
  const APInt& right = rhs->getValue();
  bool result;
  switch (compare.getPredicate()) {
    case CmpInst::ICMP_EQ:
      result = left == right;
      break;
    case CmpInst::ICMP_NE:
      result = left != right;
      break;
    case CmpInst::ICMP_UGT:
      result = left.ugt(right);
      break;
    case CmpInst::ICMP_UGE:
      result = left.uge(right);
      break;
    case CmpInst::ICMP_ULT:
      result = left.ult(right);
      break;
    case CmpInst::ICMP_ULE:
      result = left.ule(right);
      break;
    case CmpInst::ICMP_SGT:
      result = left.sgt(right);
      break;
    case CmpInst::ICMP_SGE:
      result = left.sge(right);
      break;
    case CmpInst::ICMP_SLT:
      result = left.slt(right);
      break;
    case CmpInst::ICMP_SLE:
      result = left.sle(right);
      break;
    default:
      return nullptr;
  }
  return ConstantInt::get(compare.getType(), result);
}

Constant*
foldCast(CastInst& cast)
{
  auto* source = dyn_cast<ConstantInt>(cast.getOperand(0));
  auto* targetType = dyn_cast<IntegerType>(cast.getType());
  if (!source || !targetType)
    return nullptr;

  APInt value = source->getValue();
  unsigned targetWidth = targetType->getBitWidth();
  switch (cast.getOpcode()) {
    case Instruction::Trunc:
      value = value.trunc(targetWidth);
      break;
    case Instruction::ZExt:
      value = value.zext(targetWidth);
      break;
    case Instruction::SExt:
      value = value.sext(targetWidth);
      break;
    default:
      return nullptr;
  }
  return ConstantInt::get(targetType, value);
}

Value*
foldInstruction(Instruction& inst)
{
  if (auto* binary = dyn_cast<BinaryOperator>(&inst))
    return foldBinary(*binary);
  if (auto* compare = dyn_cast<ICmpInst>(&inst))
    return foldCompare(*compare);
  if (auto* cast = dyn_cast<CastInst>(&inst))
    return foldCast(*cast);
  if (auto* select = dyn_cast<SelectInst>(&inst)) {
    auto* condition = dyn_cast<ConstantInt>(select->getCondition());
    if (condition)
      return select->getOperand(condition->isZero() ? 2 : 1);
  }
  return nullptr;
}

} // namespace

PreservedAnalyses
ConstantFolding::run(Module& mod, ModuleAnalysisManager& mam)
{
  int constFoldTimes = 0;

  for (Function& function : mod)
    for (BasicBlock& block : function)
      for (auto it = block.begin(), end = block.end(); it != end;) {
        Instruction& inst = *it++;
        Value* folded = foldInstruction(inst);
        if (!folded)
          continue;
        inst.replaceAllUsesWith(folded);
        inst.eraseFromParent();
        ++constFoldTimes;
      }

  mOut << "ConstantFolding running...\nTo eliminate " << constFoldTimes
       << " instructions\n";
  return constFoldTimes ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
