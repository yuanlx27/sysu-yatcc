# YatCC: Yat Compiler Course

This course is the practical companion to Compiler Principles. Through hands-on compiler construction, students learn the principles, methods, techniques, and tools used to implement programming languages and domain-specific languages. The coursework covers lexical analysis with handwritten scanners and Lex-family tools, top-down and bottom-up parsing, Yacc-family tools, semantic analysis, intermediate code generation, and problem-specific semantic actions.

See the [original repository](https://github.com/arcsysu/YatCC) for details.

## Setup

Install [Docker](https://docs.docker.com/engine/install) and the [Dev Container CLI](https://github.com/devcontainers/cli), then start the development container from the repository root:

```bash
devcontainer up
devcontainer exec bash
```

Inside the container, configure the project and run Task 0 to verify the environment:

```bash
cmake -B build && cmake --build build --target task0-score
```

Full score means the environment is set up correctly.
