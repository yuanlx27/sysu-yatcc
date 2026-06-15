#pragma once

#include <llvm/IR/PassManager.h>

class ConstantPropagation : public llvm::PassInfoMixin<ConstantPropagation>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class AlgebraicSimplification
  : public llvm::PassInfoMixin<AlgebraicSimplification>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class StrengthReduction : public llvm::PassInfoMixin<StrengthReduction>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class InstructionCombining : public llvm::PassInfoMixin<InstructionCombining>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class CommonSubexpressionElimination
  : public llvm::PassInfoMixin<CommonSubexpressionElimination>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class LoopInvariantCodeMotion
  : public llvm::PassInfoMixin<LoopInvariantCodeMotion>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class DeadCodeElimination : public llvm::PassInfoMixin<DeadCodeElimination>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};

class ControlFlowSimplification
  : public llvm::PassInfoMixin<ControlFlowSimplification>
{
public:
  llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager&);
};
