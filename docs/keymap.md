# VertexFEM: HPC Architecture, Scientific Computing Foundations, and Literature Keymap

> **Literature Search & Compliance Notification**:  
> This architectural keymap is constructed based on seminal and state-of-the-art literature retrieved and verified via the `literature-search-arxiv` skill.  
> Users and developers must review the [arXiv API Terms of Use](https://info.arxiv.org/help/api/index.html) and comply with individual paper license agreements (recorded in [.licenses/literature_search_arxiv_LICENSE.txt](file:///home/sallyface/project/c/VertexFEM/.licenses/literature_search_arxiv_LICENSE.txt)).

---

## 1. Literature Repository & Metadata Index

The full-text PDF documents are archived locally under the `docs/arxiv/` directory as foundational references for VertexFEM system engineering:

| arXiv Identifier | Title | Authors / Primary Organization | Local Archive Path | Official Web Citation |
| :--- | :--- | :--- | :--- | :--- |
| **[2601.08374](https://arxiv.org/abs/2601.08374)** | *Shifting the Sweet Spot: High-Performance Matrix-Free Method for High-Order Elasticity* (Jan 2026) | MFEM Team / Lawrence Livermore National Laboratory | [PDF](file:///home/sallyface/project/c/VertexFEM/docs/arxiv/2601.08374_mfem_matrix_free_elasticity.pdf) | [arXiv:2601.08374](https://arxiv.org/abs/2601.08374) |
| **[2109.04996](https://arxiv.org/abs/2109.04996)** | *Efficient Exascale Discretizations: High-Order Finite Element Methods* (2021) | CEED Co-Design Team (Kolev, Fischer, Dongarra, Brown, et al.) | [PDF](file:///home/sallyface/project/c/VertexFEM/docs/arxiv/2109.04996_efficient_exascale_discretizations.pdf) | [arXiv:2109.04996](https://arxiv.org/abs/2109.04996) |
| **[1711.03590](https://arxiv.org/abs/1711.03590)** | *Fast matrix-free evaluation of discontinuous Galerkin finite element operators* (2019) | Martin Kronbichler & Katharina Kormann (deal.II / TUM) | [PDF](file:///home/sallyface/project/c/VertexFEM/docs/arxiv/1711.03590_matrix_free_dg_operators.pdf) | [arXiv:1711.03590](https://arxiv.org/abs/1711.03590) |
| **[1911.09220](https://arxiv.org/abs/1911.09220)** | *MFEM: A modular finite element methods library* (2020) | Robert Anderson, Veselin Dobrev, Tzanio Kolev, et al. (LLNL) | [PDF](file:///home/sallyface/project/c/VertexFEM/docs/arxiv/1911.09220_mfem_modular_fem_library.pdf) | [arXiv:1911.09220](https://arxiv.org/abs/1911.09220) |

---

## 2. Core Theoretical Foundations & HPC Architectural Synthesis

```
       ┌─────────────────────────────────────────────────────────────┐
       │             Exascale Bottleneck: FLOPs are Cheap, Data is Costly            │
       │   Full Assembly (FA):    ~15 Bytes/FLOP (Memory-bandwidth starved)  │
       │   Partial Assembly (PA): ~1 Byte/FLOP   (Saturating compute cores)  │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│  CEED Operator Pipeline              │  │  Matrix-Free Sum-Factorization       │
│  (arXiv:2109.04996)                  │  │  (arXiv:1711.03590 / 2601.08374)     │
│  G: Subdomain Restriction (Halo/DOF) │  │  3D Operator Eval: O(p^6) -> O(p^4)  │
│  B: 1D Tensor Basis Interpolation    │  │  Even-Odd Symmetry: 50% FLOP Reduction│
│  D: Pointwise Physics / Quadrature   │  │  SIMD Cell-Batching & Kernel Fusion  │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│  Dual-Memory State Machine           │  │  Matrix-Free Solvers & Preconditioners│
│  (arXiv:1911.09220)                  │  │  (arXiv:2601.08374)                  │
│  5 States: HOST, DEVICE, SYNC, etc.  │  │  Matrix-Free Chebyshev Smoothers     │
│  Demand-driven, Zero-Overhead Pinning│  │  p-Multigrid with Coarse AMG Fallback│
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```

### 2.1 The CEED Operator Decomposition Pipeline ($A = G^T B^T D B G$)
As established by the Center for Efficient Exascale Discretizations (arXiv:2109.04996), legacy FEM frameworks bind mesh topology, quadrature integration, and physical constitutive laws into a monolithic `computeLocalStiffness()` routine. In contrast, CEED factors any global finite element operator $A$ into four orthogonal operators:

$$A = G^T B^T D B G$$

1. **$G$ (Subdomain Restriction & Assembly / Scatter-Gather Operator)**:
   Maps a global degrees-of-freedom vector ($E$-vector, shared across processes and partitions) to a sub-vector of duplicated element degrees of freedom ($L$-vector). It encapsulates all parallel MPI boundary exchanges and local permutation tables.
2. **$B$ (Basis Evaluation / Interpolation Operator)**:
   Transforms element degrees of freedom ($L$-vector) into field values and physical gradients evaluated at quadrature points ($Q$-vector). Crucially, for tensor-product elements (quads/hexes), $B$ is expressed as a Kronecker product of 1D interpolation matrices, completely decoupled from PDE-specific physics.
3. **$D$ (Pointwise Physics Operator)**:
   Evaluates constitutive equations, material jacobians, and geometric transformation metric tensors ($J^{-1} \det(J)$) at individual quadrature points. $D$ is strictly local (diagonal or block-diagonal across quadrature points), with zero inter-element coupling.
4. **$B^T$ and $G^T$ (Adjoint Basis Action & Global Accumulation)**:
   Transposes test space operations back onto the dual global vector space.

---

### 2.2 Sum-Factorization & Microarchitectural Optimizations
In high-order discretizations on $d$-dimensional tensor-product hexahedral elements ($d=3$):
* **Algorithmic Complexity Reduction**:
  Full sparse matrix assembly requires storing and computing $\mathcal{O}(p^{2d}) = \mathcal{O}(p^6)$ non-zero entries per element. Sum-factorization contracts tensor dimensions sequentially ($x \to y \to z$), reducing the operation count to:
  $$\mathcal{O}(d \cdot p^{d+1}) = \mathcal{O}(p^4)$$
* **Even-Odd Symmetry Decomposition (Kronbichler & Kormann, arXiv:1711.03590)**:
  Exploiting the reflectional symmetry and antisymmetry of 1D Lagrange interpolants around the element reference origin ($[-1, 1]$), the 1D interpolation matrix $I$ and differentiation matrix $D$ split into symmetric and skew-symmetric sub-blocks. This cuts 1D matrix-vector multiplication operations by **exactly 50%**.
* **SIMD Cell-Batching (CPU Vectorization)**:
  Rather than processing elements individually (which suffers from short vector register lengths at modest $p$), independent non-adjacent elements are packed across SIMD vector lanes (AVX2 / AVX-512 / ARM SVE). Each lane computes an independent element, achieving $>60\%$ of hardware peak arithmetic throughput.
* **Macro-Kernel Fusion & Slice-Wise Loop Reorganization (MFEM 2026, arXiv:2601.08374)**:
  For vector-valued systems (e.g., 3D linear elasticity with $(u_x, u_y, u_z)$ DOFs), loop structures are rearranged into 2D slice-wise buffers that reside in L1/L2 caches and GPU Shared Memory. Combined with Voigt symmetry, this pushes the performance sweet spot of matrix-free evaluation from low orders ($p \approx 2$) up to high orders ($p \ge 6$), yielding **7×–83× core speedups**.

---

### 2.3 Heterogeneous Memory Abstraction & 5-State Machine (arXiv:1911.09220)
Modern GPU architectures (NVIDIA H100, RTX 4060, AMD MI300) provide high HBM/GDDR bandwidth (2.0–3.3 TB/s), whereas the host-device PCIe link is bounded by 32–64 GB/s. MFEM’s memory architecture addresses this via:
1. **5-State Coherence Model**:
   `EMPTY`, `UNINITIALIZED`, `HOST_VALID`, `DEVICE_VALID`, `SYNCHRONIZED`. Data resides persistently on the device; Host-to-Device (H2D) and Device-to-Host (D2H) copies are strictly lazy and demand-driven.
2. **Container Pinning via RAII `AccessGuard`**:
   Ensures that during mathematical kernel execution (e.g., Krylov iterations), the underlying memory records are locked against structural reallocation (`resize`, `reserve`, or move).

---

### 2.4 Matrix-Free Solvers & $p$-Multigrid Preconditioning
Because matrix-free methods never assemble explicit sparse matrices (CSR/ELL), classical algebraic multigrid (AMG) cannot be directly applied:
* **Chebyshev Polynomial Smoothers**:
  Matrix-free operator evaluation is integrated into high-order Chebyshev polynomial smoothers. These require only matrix-vector products ($y = Ax$) and estimates of the extremal eigenvalues of $D^{-1}A$, avoiding explicit matrix entries.
* **$p$-Multigrid / Geometric Multigrid (GMG)**:
  Hierarchical coarsening is performed in polynomial order: $p \to p-1 \to \dots \to 1$.
* **Coarse AMG Fallback**:
  Only at the coarsest level ($p=1$) is a compact sparse matrix assembled and handed off to external solvers (e.g., HYPRE `BoomerAMG`).

---

## 3. Direct Architectural Keymap to VertexFEM Codebase

The following matrix maps the literature foundations directly to current and planned VertexFEM header and source modules:

| Theoretical / Algorithmic Pillar | Literature Reference | VertexFEM Target Module | Role, Architectural Boundary, and Implementation Standards |
| :--- | :--- | :--- | :--- |
| **5-State Memory Coherence** | arXiv:1911.09220 | [mem_manager.hpp](file:///home/sallyface/project/c/VertexFEM/include/base/mem_manager.hpp)<br>[mem_manager.cpp](file:///home/sallyface/project/c/VertexFEM/src/base/mem_manager.cpp) | Implements heterogeneous state machine; manages lazy synchronizations, avoiding PCIe thrashing. |
| **Container Pinning (AccessGuard)** | docs/analysis.md | [array.hpp](file:///home/sallyface/project/c/VertexFEM/include/base/array.hpp) | Atomic reader-writer leases; locks container structure during compute kernels. |
| **CUDA Runtime & Async Streams** | arXiv:1911.09220 | [cuda.hpp](file:///home/sallyface/project/c/VertexFEM/include/base/cuda.hpp)<br>[cuda.cpp](file:///home/sallyface/project/c/VertexFEM/src/base/cuda.cpp) | Auto-registered into `MemoryManager`; non-blocking streaming execution. |
| **CEED $G$ Restriction Operator** | arXiv:2109.04996 | [mesh.hpp](file:///home/sallyface/project/c/VertexFEM/include/mesh/mesh.hpp)<br>[fespace.hpp](file:///home/sallyface/project/c/VertexFEM/include/fem/fespace.hpp) | Combinatorial entity tables (`E->V`, `E->F`); non-blocking MPI halo exchange. |
| **CEED $B$ Basis Action (Sum-Fact)** | arXiv:1711.03590<br>arXiv:2601.08374 | [fespace.hpp](file:///home/sallyface/project/c/VertexFEM/include/fem/fespace.hpp)<br>[assembler.hpp](file:///home/sallyface/project/c/VertexFEM/include/asm/assembler.hpp) | 1D GLL tensor-product rules; Even-Odd decomposition; CPU SIMD batching & GPU Shared Memory contraction. |
| **CEED $D$ Pointwise Physics Kernels** | arXiv:2109.04996 | [assembler.hpp](file:///home/sallyface/project/c/VertexFEM/include/asm/assembler.hpp) | Stores quadrature geometric metric factors ($J^{-1}\det(J)$); fuses material constitutive laws into $B^T$. |
| **Abstract Matrix-Free Operator** | arXiv:2109.04996 | [vector.hpp](file:///home/sallyface/project/c/VertexFEM/include/linalg/vector.hpp) | C++20 `LinearOperator` concept exposing zero-allocation `apply(x, y)` virtual and concept interfaces. |
| **Chebyshev Smoothers & Krylov Solvers** | arXiv:2601.08374 | [linear_solver.hpp](file:///home/sallyface/project/c/VertexFEM/include/solver/linear_solver.hpp) | Matrix-free `CGSolver`, `GMRESSolver`, and polynomial smoothers for $p$-multigrid hierarchies. |
| **Pimpl External Solver ABI Firewall** | docs/analysis.md | [linear_solver.hpp](file:///home/sallyface/project/c/VertexFEM/include/solver/linear_solver.hpp)<br>`src/solver/hypre_amg.cpp` | Pimpl wrapper shielding external C headers (`HYPRE.h`); acts as the $p=1$ coarse AMG solver. |

---

## 4. Gap Diagnostics & Five-Stage Actionable Implementation Roadmap

Comparing VertexFEM's current baseline against the state of the art, the primary engineering challenge is bridging **Layer 0 (Heterogeneous Memory & Containers)** with **Layers 1–5 (Mathematical Operators, Discretization, and Solvers)**.

The following roadmap translates the literature principles into prioritized development milestones:

### Stage 1: Layer 0 Hardening & Linear Algebra Foundation (Milestone 1)
1. **Automate CUDA Backend Registration**:
   Mount `cudaAllocDev`, `cudaAllocPin`, `cudaAllocMan`, `cudaCopyHtoD`, and `cudaCopyDtoH` automatically inside `vfem::initialize()`, resolving device allocation fallback to Host memory.
2. **Guarantee 64-Byte Alignment**:
   Update `MemoryManager::requiredAlignment` to enforce a minimum of 64 bytes for `MemType::HOST`, preventing AVX-512 `#GP` faults on aligned vector loads.
3. **Implement Core `Vector<T>`**:
   Build [include/linalg/vector.hpp](file:///home/sallyface/project/c/VertexFEM/include/linalg/vector.hpp) atop `Array<T>`. Implement BLAS-1 kernels (`dot`, `norm`, `axpy`) using SIMD-vectorized direct pointer views, completely bypassing the scalar state machine in inner loops.

### Stage 2: Topological Mesh & Parallel Restriction $G$ (Milestone 2)
1. **Gmsh MSH 4.1 Parser**:
   Parse high-order quad and hex meshes into flat, contiguous combinatorial tables.
2. **Halo Exchange Engine (CEED $G$)**:
   Implement non-blocking point-to-point MPI communication (`MPI_Irecv` / `MPI_Isend`) to synchronize shared boundary degrees of freedom between distributed subdomains.

### Stage 3: High-Order Discretization & Tensor-Product Basis $B$ (Milestone 3)
1. **1D Quadrature & Polynomial Spaces**:
   Implement Gauss-Legendre and Gauss-Lobatto-Legendre rules up to order $p=12$.
2. **Even-Odd Kernel Decomposition**:
   Decompose 1D interpolation matrices into symmetric and skew-symmetric sub-matrices as detailed in Kronbichler (arXiv:1711.03590), halving 1D contraction operations.

### Stage 4: Matrix-Free Sum-Factorization Assembly Engine (Milestone 4)
1. **CPU SIMD Cell-Batching**:
   Process 4 (AVX2) or 8 (AVX-512) hex elements concurrently across SIMD vector lanes.
2. **GPU Shared Memory Kernels**:
   Implement CUDA kernels storing 1D basis tables in Shared Memory, performing sequential $x \to y \to z$ contraction passes with block-level barrier synchronizations.
3. **Fused Metric Physics ($D$)**:
   Store quadrature metrics compactly and evaluate physical fluxes directly within the basis transformation pipeline.

### Stage 5: Matrix-Free Krylov Solvers & Multigrid Preconditioning (Milestone 5)
1. **Matrix-Free CG & GMRES**:
   Deploy native Krylov solvers operating transparently over `LinearOperator` interfaces.
2. **Matrix-Free Chebyshev Smoothers**:
   Implement polynomial smoothers using power-method eigenvalue estimation on matrix-free operators.
3. **Pimpl HYPRE BoomerAMG Integration**:
   Isolate HYPRE headers behind ABI firewalls and apply BoomerAMG as the coarse-level solver on low-order ($p=1$) grids.
