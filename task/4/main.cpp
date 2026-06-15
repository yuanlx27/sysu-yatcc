#include <cstdlib>
#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

#include "ConstantFolding.hpp"
#include "Mem2Reg.hpp"
#include "OptimizationPasses.hpp"
#include "StaticCallCounter.hpp"
#include "StaticCallCounterPrinter.hpp"

#ifdef TASK4_LLM

#include <pybind11/embed.h>

#include "PassSequencePredict.hpp"

namespace Py = pybind11;

#endif

void
addOptimizationPipeline(llvm::ModulePassManager& mpm)
{
  using namespace llvm;

  mpm.addPass(Mem2Reg());
  mpm.addPass(ConstantPropagation());

  // Early scalar cleanup exposes common expressions and loop invariants.
  mpm.addPass(ConstantFolding(errs()));
  mpm.addPass(AlgebraicSimplification());
  mpm.addPass(InstructionCombining());
  mpm.addPass(StrengthReduction());
  mpm.addPass(CommonSubexpressionElimination());

  mpm.addPass(LoopInvariantCodeMotion());

  // Hoisting and CSE create new folding and dead-code opportunities.
  mpm.addPass(ConstantFolding(errs()));
  mpm.addPass(AlgebraicSimplification());
  mpm.addPass(InstructionCombining());
  mpm.addPass(CommonSubexpressionElimination());
  mpm.addPass(DeadCodeElimination());
  mpm.addPass(ControlFlowSimplification());
  mpm.addPass(AlgebraicSimplification());
  mpm.addPass(DeadCodeElimination());
}

void
opt(llvm::Module& mod)
{
  using namespace llvm;

#ifdef TASK4_LLM
  std::unique_ptr<Py::scoped_interpreter> python;
#endif

  // 定义分析pass的管理器
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  ModulePassManager mpm;

  // 注册分析pass的管理器
  PassBuilder pb;
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // 添加分析pass到管理器中
  mam.registerPass([]() { return StaticCallCounter(); });

#ifdef TASK4_LLM

  const char* apiKey = std::getenv("YATCC_LLM_API_KEY");
  const char* baseURL = std::getenv("YATCC_LLM_BASE_URL");
  if (!apiKey || !*apiKey || !baseURL || !*baseURL) {
    llvm::errs() << "YATCC_LLM_API_KEY/YATCC_LLM_BASE_URL are not configured; "
                    "using the deterministic optimization pipeline.\n";
    addOptimizationPipeline(mpm);
  } else {
    try {
      python = std::make_unique<Py::scoped_interpreter>();
      Py::module_ sys = Py::module_::import("sys");
      sys.attr("path").attr("append")(TASK4_DIR);

      mpm.addPass(PassSequencePredict(
        apiKey,
        baseURL,
        {
          { "Mem2Reg",
            TASK4_DIR "/Mem2Reg.hpp",
            TASK4_DIR "/Mem2Reg.cpp",
            "Mem2Reg.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(Mem2Reg());
            } },
          { "ConstantPropagation",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "ConstantPropagation.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(ConstantPropagation());
            } },
          { "ConstantFolding",
            TASK4_DIR "/ConstantFolding.hpp",
            TASK4_DIR "/ConstantFolding.cpp",
            "ConstantFolding.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(ConstantFolding(llvm::errs()));
            } },
          { "AlgebraicSimplification",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "AlgebraicSimplification.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(AlgebraicSimplification());
            } },
          { "InstructionCombining",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "InstructionCombining.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(InstructionCombining());
            } },
          { "StrengthReduction",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "StrengthReduction.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(StrengthReduction());
            } },
          { "CommonSubexpressionElimination",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "CommonSubexpressionElimination.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(CommonSubexpressionElimination());
            } },
          { "LoopInvariantCodeMotion",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "LoopInvariantCodeMotion.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(LoopInvariantCodeMotion());
            } },
          { "DeadCodeElimination",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "DeadCodeElimination.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(DeadCodeElimination());
            } },
          { "ControlFlowSimplification",
            TASK4_DIR "/OptimizationPasses.hpp",
            TASK4_DIR "/OptimizationPasses.cpp",
            "ControlFlowSimplification.xml",
            [](llvm::ModulePassManager& passes) {
              passes.addPass(ControlFlowSimplification());
            } },
        }));
    } catch (const Py::error_already_set& error) {
      llvm::errs() << "Unable to initialize the LLM optimizer; using the "
                      "deterministic pipeline: "
                   << error.what() << '\n';
      addOptimizationPipeline(mpm);
    }
  }

#else

  mpm.addPass(StaticCallCounterPrinter(llvm::errs()));
  addOptimizationPipeline(mpm);

#endif

  // 运行优化pass
  mpm.run(mod, mam);
}

int
main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <input> <output>\n";
    return -1;
  }

  llvm::LLVMContext ctx;

  llvm::SMDiagnostic err;
  auto mod = llvm::parseIRFile(argv[1], err, ctx);
  if (!mod) {
    std::cout << "Error: unable to parse input file: " << argv[1] << '\n';
    err.print(argv[0], llvm::errs());
    return -2;
  }

  std::error_code ec;
  llvm::StringRef outPath(argv[2]);
  llvm::raw_fd_ostream outFile(outPath, ec);
  if (ec) {
    std::cout << "Error: unable to open output file: " << argv[2] << '\n';
    return -3;
  }

  opt(*mod); // IR的优化发生在这里

  mod->print(outFile, nullptr, false, true);
  if (llvm::verifyModule(*mod, &llvm::outs()))
    return 3;
}
