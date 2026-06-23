# VertexFEM

**Vertex-centered Finite Element Method library — work in progress.**

VertexFEM is a high-performance C++20 library for vertex-centered finite element methods, targeting both CPU and GPU architectures with MPI-based distributed parallelism.

> **Status:** Early development (v0.1.0). Core APIs are unstable and substantial modules are still being implemented. Not ready for production use.

## Features (planned / in progress)

- **Vertex-centered FEM formulation** — unknowns reside at mesh vertices, enabling natural coupling with vertex-based discretizations
- **Multi-backend GPU support** — CUDA, HIP, and SYCL via unified `__host__ __device__` abstractions
- **Distributed memory** — MPI parallelism with optional solvers: PETSc, SLEPc, MUMPS, SuperLU_DIST
- **Dual precision** — compile-time selection of `float` or `double` via `VFEM_USE_DOUBLE` / `VFEM_USE_SINGLE`
- **Flexible memory management** — host, device, managed, and pinned allocations with Umpire integration
- **Modular design** — mesh, finite element spaces, assembly, linear algebra, solvers, and I/O as independent modules

## Requirements

- **Compiler:** GCC 11+ or Clang 14+ with C++20 support
- **Build system:** CMake ≥ 3.14
- **Optional dependencies:**
  - MPI (for distributed parallelism)
  - CUDA / HIP / SYCL (for GPU offload)
  - PETSc / SLEPc / MUMPS / SuperLU_DIST (for advanced solvers)
  - Umpire (for memory pool management)

## Quick start

```sh
git clone https://github.com/VertexFEM/VertexFEM.git
cd VertexFEM
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Build options

| Option | Default | Description |
|--------|---------|-------------|
| `VERTEXFEM_USE_DOUBLE` | `ON` | Use double precision; disable for single precision |
| `VERTEXFEM_BUILD_TESTS` | `ON` | Build the test / benchmark harness |

## Project structure

```
VertexFEM/
├── config/          # Compile-time configuration & platform detection
├── include/         # Public headers
│   ├── asm/         #   Assembly routines
│   ├── base/        #   Memory management & error handling
│   ├── fem/         #   Finite element spaces
│   ├── io/          #   Mesh I/O
│   ├── linalg/      #   Linear algebra
│   ├── mesh/        #   Mesh data structures
│   ├── solver/      #   Linear & nonlinear solvers
│   └── utils/       #   Utilities & benchmarking
├── src/             # Implementation files (mirrors include/)
├── test/            # Tests & microbenchmarks
├── data/            # Sample mesh files
└── 3party/          # Third-party dependencies
```

## License

This project is licensed under the Academic Free License v3.0. See [LICENSE](LICENSE) for details.
