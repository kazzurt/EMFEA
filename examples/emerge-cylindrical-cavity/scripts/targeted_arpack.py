"""Optional, benchmark-specific replacement for EMerge's ARPACK eigensolver.

Scope: the 100 mm diameter x 100 mm long, air-filled, entirely PEC cylinder
in solve_cavity.py, requesting ten modes around exactly 3 GHz. This is not a
general Maxwell eigensolver or an upstream EMerge change.

At this shift the analytic cylinder spectrum has 16 physical eigenvectors
closer to the shift in squared wavenumber than the gradient nullspace. EMerge
3.0.0b6 requests 30 eigenpairs for ten requested modes. This adapter requests
14 instead, with ncv=56, to avoid deliberately searching that nullspace
cluster. It retains the upstream SuperLU options, shift-invert mapping,
ARPACK tolerance, and final sorting rule. Whether this improves runtime must
be measured; the code does not establish that LU or nullspace convergence
caused any previous long run.

After constructing the benchmark model, enable explicitly with:

    from targeted_arpack import BENCHMARK_ID, TargetedCylinderARPACK
    model.mw.solveroutine.set_solver(
        TargetedCylinderARPACK(benchmark_id=BENCHMARK_ID)
    )

The identifier is an explicit caller attestation of the geometry/material/PEC
scope; the matrix alone
cannot establish those properties. Keep the default EMerge run separately
for comparison. Installed package files are not modified.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import psutil
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

from emerge._emerge.solver import EigSolver


BENCHMARK_ID = "air-pec-cylinder-d100mm-l100mm-f3GHz"
_C0 = 299_792_458.0


class TargetedCylinderARPACK(EigSolver):
    """Reduced-oversampling ARPACK for this one lossless cylinder benchmark."""

    name = "TargetedCylinderARPACK-k14-ncv56"

    def __init__(
        self,
        pre: str = "targeted-cylinder",
        *,
        benchmark_id: str,
        residual_tolerance: float = 1e-6,
        maxiter: int | None = None,
    ) -> None:
        super().__init__(pre)
        if benchmark_id != BENCHMARK_ID:
            raise ValueError("This adapter is restricted to " + BENCHMARK_ID)
        if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be positive and finite")
        if maxiter is not None and (not isinstance(maxiter, int) or maxiter <= 0):
            raise ValueError("maxiter must be a positive integer or None")
        self.benchmark_id = benchmark_id
        self.residual_tolerance = residual_tolerance
        # None preserves SciPy's default, as in EMerge 3.0.0b6. A parent process
        # can impose a wall-clock limit; this adapter creates no background job.
        self.maxiter = maxiter
        self.last_report: dict = {}

    def duplicate(self) -> "TargetedCylinderARPACK":
        return type(self)(
            pre=self.pre,
            benchmark_id=self.benchmark_id,
            residual_tolerance=self.residual_tolerance,
            maxiter=self.maxiter,
        )

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0.0,
        which: str = "LM",
        sign: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (issparse(A) and issparse(B)):
            raise TypeError("A and B must be sparse matrices")
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A and B must be square matrices of equal shape")
        if nmodes != 10 or which != "LM" or sign != 1.0:
            raise ValueError("This benchmark requires nmodes=10, which='LM', sign=1")
        search_hz = float(target_k0) * _C0 / (2.0 * math.pi)
        if not math.isfinite(search_hz) or abs(search_hz - 3e9) > 1.0:
            raise ValueError("This adapter has only been scoped for a 3 GHz shift")

        k_request, ncv = 14, 56
        n_total = A.shape[0]
        if not (nmodes < k_request < ncv < n_total):
            raise ValueError("Need 10 < k_request=14 < ncv=56 < matrix dimension")
        for label, matrix in (("A", A), ("B", B)):
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{label} contains nonfinite entries")
            scale = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
            imaginary = float(np.max(np.abs(matrix.data.imag))) if matrix.nnz else 0.0
            if scale == 0.0:
                raise ValueError(f"{label} is zero")
            if imaginary > 1e-13 * scale:
                raise ValueError(f"{label} is materially complex; lossless PEC scope required")

        started = time.perf_counter()
        process = psutil.Process(os.getpid())

        def event(stage: str, **extra) -> None:
            print(
                "TARGETED_ARPACK "
                + json.dumps(
                    {
                        "event": stage,
                        "elapsed_s": time.perf_counter() - started,
                        "rss_gib": process.memory_info().rss / 1024**3,
                        **extra,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        sigma = (sign * target_k0**2).real
        shifted = (A - sigma * B).tocsc()
        event(
            "before_lu", n_dof=n_total, a_nnz=int(A.nnz),
            b_nnz=int(B.nnz), shifted_nnz=int(shifted.nnz),
            k_request=k_request, ncv=ncv, target_frequency_hz=search_hz,
        )
        lu_start = time.perf_counter()
        lu = splu(
            shifted,
            permc_spec="MMD_AT_PLUS_A",
            relax=0,
            diag_pivot_thresh=0.001,
            options={"SymmetricMode": True, "Equil": False, "IterRefine": "SINGLE"},
        )
        lu_seconds = time.perf_counter() - lu_start
        event("after_lu", stage_s=lu_seconds)

        matvec_calls = 0

        def shift_invert_matvec(vector: np.ndarray) -> np.ndarray:
            nonlocal matvec_calls
            matvec_calls += 1
            return lu.solve(B @ vector)

        operator = LinearOperator(
            A.shape, matvec=shift_invert_matvec, dtype=shifted.dtype
        )
        event("before_arpack", tolerance=1e-8, maxiter=self.maxiter)
        arpack_start = time.perf_counter()
        try:
            transformed, vectors = eigs(
                operator, k=k_request, which=which, ncv=ncv,
                tol=1e-8, maxiter=self.maxiter,
            )
        except Exception as exc:
            event(
                "arpack_failed", stage_s=time.perf_counter() - arpack_start,
                matvec_calls=matvec_calls, error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        arpack_seconds = time.perf_counter() - arpack_start
        event("after_arpack", stage_s=arpack_seconds, matvec_calls=matvec_calls)
        if transformed.shape != (k_request,) or vectors.shape != (n_total, k_request):
            raise RuntimeError("ARPACK returned an unexpected eigenpair count or shape")
        if not np.all(np.isfinite(transformed)) or np.any(transformed == 0):
            raise RuntimeError("ARPACK returned invalid transformed eigenvalues")

        # Identical eigenvalue back-mapping and final sort to upstream EMerge.
        eigenvalues = sigma + 1.0 / transformed
        wavenumbers = np.sqrt(eigenvalues / sign)
        selected = np.argsort(np.abs(wavenumbers.real - target_k0))[:nmodes]
        values = eigenvalues[selected]
        modes = vectors[:, selected]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(modes)):
            raise RuntimeError("Selected eigenpairs contain nonfinite values")

        residuals = []
        for value, vector in zip(values, modes.T):
            av, bv = A @ vector, B @ vector
            denominator = np.linalg.norm(av) + abs(value) * np.linalg.norm(bv)
            if denominator == 0.0:
                raise RuntimeError("Selected an eigenvector with zero residual scale")
            residual = float(np.linalg.norm(av - value * bv) / denominator)
            residuals.append(residual)

        self.last_report = {
            "benchmark_id": BENCHMARK_ID,
            "k_request": k_request,
            "ncv": ncv,
            "n_selected": len(values),
            "lu_seconds": lu_seconds,
            "arpack_seconds": arpack_seconds,
            "matvec_calls": matvec_calls,
            "relative_generalized_residuals": residuals,
            "residual_tolerance": self.residual_tolerance,
        }
        event("residual_check", **self.last_report)
        if not all(math.isfinite(r) and r <= self.residual_tolerance for r in residuals):
            raise RuntimeError("Selected eigenpairs failed the generalized residual check")
        event("complete")
        return values, modes
