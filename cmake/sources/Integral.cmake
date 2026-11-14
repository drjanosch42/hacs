# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Integral/ added by slots 9..26 (S24 CL-10, CL-10b, S25 FOLD, S26 FOLD; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeBredonTreeRingValidation.cpp
    src/hacs/Integral/Model/IsotropyModel.cpp
    src/hacs/Integral/HeckeIsotropyModelValidation.cpp
    src/hacs/Integral/Model/IntegralGroups.cpp
    src/hacs/Integral/HeckeFreeModelValidation.cpp
    src/hacs/Integral/Model/StrictLift.cpp
    src/hacs/Integral/HeckeStrictLiftValidation.cpp
    src/hacs/Integral/Model/IntegralHecke.cpp
    src/hacs/Integral/HeckeIntegralActionValidation.cpp
    src/hacs/Integral/Model/PeriodicityClass.cpp
    src/hacs/Integral/HeckePeriodicityClassValidation.cpp
    src/hacs/Integral/HeckeStrictLiftRelationsValidation.cpp
    src/hacs/Integral/Model/IsotropyModelEll.cpp
    src/hacs/Integral/HeckeIsotropyModelEllValidation.cpp
    src/hacs/Integral/HeckeIsotropyModelEllOpenHalvesValidationArms.cpp
    src/hacs/Integral/Model/RefinedStrictLift.cpp
    src/hacs/Integral/HeckeIsotropyModelEllStrictRefinedValidationArms.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 DESCENT0
    src/hacs/Integral/Descent/DescentAssembly.cpp
    src/hacs/Integral/Descent/DescentOracle.cpp
    src/hacs/Integral/Descent/IntegralDescentGroupsValidation.cpp
    # --- slot 24 DESCENT1
    src/hacs/Integral/Descent/DescentLevelN.cpp
    src/hacs/Integral/Descent/IntegralDescentLevelNValidation.cpp
    # --- slot 24 DESCENT2
    src/hacs/Integral/Descent/DescentMaps.cpp
    src/hacs/Integral/Descent/IntegralDescentMapsValidation.cpp
    # --- slot 24 DESCENT4
    src/hacs/Integral/Descent/DescentPair.cpp
    src/hacs/Integral/Descent/DescentPairHecke.cpp
    src/hacs/Integral/Descent/IntegralDescentPairValidation.cpp
    # --- slot 24 DESCENT5
    src/hacs/Integral/Model/TreeCochains.cpp
    src/hacs/Integral/Model/IsotropyModelTree.cpp
    src/hacs/Integral/Descent/DescentTree.cpp
    src/hacs/Integral/Descent/TreeComparisonArms.cpp
    src/hacs/Integral/Descent/HeckeIsotropyModelTreeValidation.cpp
    # --- slot 24 DESCENT3
    src/hacs/Integral/Descent/DescentHeckeCrt.cpp
    src/hacs/Integral/Descent/DescentHeckeOps.cpp
    src/hacs/Integral/Descent/DescentHeckeTorsion.cpp
    src/hacs/Integral/Descent/IntegralDescentHeckeValidation.cpp
    # --- slot 24 DESCENT5c
    src/hacs/Integral/Model/TreePair.cpp
    src/hacs/Integral/Descent/DescentTreePair.cpp
    src/hacs/Integral/Descent/IntegralDescentTreePairValidation.cpp
    # --- slot 24 DESCENT6
    src/hacs/Integral/Descent/DescentRecord.cpp
    src/hacs/Integral/Descent/IntegralDescentRecordValidation.cpp
    # --- slot 24 PARD2
    src/hacs/Integral/Descent/DescentSmith.cpp
    src/hacs/Integral/Descent/DescentSmithKernels.cpp
    src/hacs/Integral/Descent/IntegralDescentSmithBlockedValidation.cpp
    # --- slot 24 PAR2K
    # DescentSmithMtValidation (standard): threaded == serial, entrywise, at T = 1, 2, 7, 16, every kernel path, real descent inputs.
    src/hacs/Integral/Descent/DescentSmithMtValidation.cpp
    # --- slot 24 PARASM
    src/hacs/Integral/Descent/DescentParAsm.cpp
    src/hacs/Integral/Descent/IntegralDescentParAsmValidation.cpp
    # --- slot 24 PARGPU
    # the ENGINE_DEVICE=gpu policy of the ring2k device LU and the installation of its operations table (static initialisation)
    src/hacs/Integral/Descent/DescentSmithGpu.cpp
    # DescentSmithGpuValidation (standard): device LU == host LU, entrywise; [SKIP] without a device.
    src/hacs/Integral/Descent/DescentSmithGpuValidation.cpp
    # --- slot 24 PARLN
    src/hacs/Integral/Descent/IntegralDescentParLnValidation.cpp
)

# Slot 25 (folded at the slot-25 close, S25 FOLD; the former cmake/slot25/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 25 HeckeZExport
    src/hacs/Integral/Descent/DescentModuleExport.cpp
    src/hacs/Integral/Descent/HeckeZExportValidation.cpp
    # --- slot 25 HeckeZRational
    # the rational step (certified side) and the untrusted factor finder (NumberField used, unchanged)
    src/hacs/Integral/HeckeZRational.cpp
    src/hacs/Integral/HeckeZFactorFinder.cpp
    # (HZ-RAT-PERF (b1)) the O(b^3) Hessenberg charpoly mod a word prime
    src/hacs/Integral/HeckeZCharpoly.cpp
    # HeckeZRationalValidation (fast)
    src/hacs/Integral/HeckeZRationalValidation.cpp
    # --- slot 25 HeckeZUTag
    src/hacs/Integral/HeckeZUTag.cpp
    # HeckeZUTagValidation (fast)
    src/hacs/Integral/HeckeZUTagValidation.cpp
    # --- slot 25 HeckeZLocal
    src/hacs/Integral/HeckeZLocal.cpp
    # HeckeZLocalValidation (standard), HeckeZLocalCensusXcheck (slow)
    src/hacs/Integral/HeckeZLocalValidation.cpp
    # --- slot 25 HeckeMazurEis
    src/hacs/Integral/HeckeMazurEis.cpp
    # HeckeMazurEisValidation (standard)
    src/hacs/Integral/HeckeMazurEisValidation.cpp
    # --- slot 25 HeckeZExportZ
    src/hacs/Integral/Descent/DescentModuleExportZ.cpp
    src/hacs/Integral/Descent/HeckeZExportZValidation.cpp
    # --- slot 25 HeckeZTor
    src/hacs/Integral/HeckeZTor.cpp
    # HeckeZTorValidation (standard)
    src/hacs/Integral/HeckeZTorValidation.cpp
    # --- slot 25 HeckeZRecord
    src/hacs/Integral/HeckeZRecord.cpp
    # HeckeZRecordValidation (standard), HeckeZCertDumpValidation (slow)
    src/hacs/Integral/HeckeZRecordValidation.cpp
    # --- slot 25 HeckeZFreeBasis
    src/hacs/Integral/Descent/IsoTreeCarrierZ.cpp
    src/hacs/Integral/Descent/HeckeZFreeBasisPairsValidation.cpp
    src/hacs/Integral/Descent/DescentFreeBasisZ.cpp
    src/hacs/Integral/Descent/HeckeZFreeBasisSatValidation.cpp
    # --- slot 25 HeckeZFreeBlock (its RNS kernels: Special.cmake, the ModArithmetic target)
    src/hacs/Integral/Descent/DescentFreeBlockCrt.cpp
    src/hacs/Integral/Descent/HeckeZFreeBlockCrtValidation.cpp
    # --- slot 25 HeckeZExportZ2
    src/hacs/Integral/Descent/DescentModuleExportZ2.cpp
    src/hacs/Integral/Descent/HeckeZExportZ2Validation.cpp
)

# Slot 26 (folded at the slot-26 close, S26 FOLD; the former cmake/slot26/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 26 HeckeZUp (Z1, UP-Z (a): U_p, p | N, over Z_(ell), ell >= 5)
    src/hacs/Integral/HeckeZUp.cpp
    src/hacs/Integral/HeckeZUpValidation.cpp
    # --- slot 26 HeckeZLocalPar (Z9 HZ-LOC-PAR): HeckeZLocalParValidation (standard)
    src/hacs/Integral/HeckeZLocalParValidation.cpp
    # --- slot 26 HeckeZEisUp (HZ-EISUP: the Eisenstein U_p systems of the full algebra at N > 1)
    src/hacs/Integral/HeckeZEisUp.cpp
    src/hacs/Integral/HeckeZEisUpValidation.cpp
    # --- slot 26 HeckeZCrossedHomOracle (Z3b: an M oracle at ell = 2, 3 independent of T~_Iso)
    src/hacs/Integral/Descent/HeckeZCrossedHomOracle.cpp
    src/hacs/Integral/Descent/HeckeZCrossedHomOracleValidation.cpp
)
