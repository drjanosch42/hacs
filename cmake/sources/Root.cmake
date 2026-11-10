# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources directly under src/hacs/ and under ModArithmetic/ (validations) added by slots 9..24 (S24 CL-10, CL-10b; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/EquivariantMappingConeModule.cpp
    src/hacs/EngineElimPassOptionsValidation.cpp
    src/hacs/EngineArenaFirstFitValidation.cpp
    src/hacs/EngineFingerprintDeviceBitValidation.cpp
    src/hacs/EngineCarriesS11Validation.cpp
    src/hacs/EngineU128CompositeValidation.cpp
    src/hacs/EngineLadicInductionValidation.cpp
    src/Engine/EngineElimSessionValidation.cpp
    src/Cuda/CudaImmaValidation.cpp
    src/Cuda/CudaImmaElimValidation.cpp
    src/ModArithmetic/Binary/BinaryF2Validation.cpp
    src/ModArithmetic/VnniGemmBitIdentityValidation.cpp
    src/hacs/HostElimThreadDeterminismValidation.cpp
    src/hacs/GpuLeanSessionValidation.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 AVX2K
    # KernelAvx2BitIdentityValidation (fast): AVX2 vs portable vs AVX-512 vs RefLinAlg, adversarial moduli / lanes / k-blocks.
    src/ModArithmetic/KernelAvx2BitIdentityValidation.cpp
    # --- slot 24 SHARD
    src/hacs/ValidationShard.cpp
)
