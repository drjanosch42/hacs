# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/HeckeExt/ added by slots 9..26 (S24 CL-10, CL-10b, S25 FOLD, S26 FOLD; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/HeckeExt/HeckeAlgebraInvariants.cpp
    src/hacs/HeckeExt/HeckeExtBindingSupport.cpp
    src/hacs/HeckeExt/HeckeExtAlgebraInvariantsValidation.cpp
    src/hacs/HeckeExt/HeckeDicksonTheta.cpp
    src/hacs/HeckeExt/HeckeDicksonThetaValidation.cpp
    src/hacs/HeckeExt/HeckeDicksonThetaHecke.cpp
    src/hacs/HeckeExt/HeckeDicksonThetaHeckeValidation.cpp
    src/hacs/HeckeExt/HeckeCensusSplit.cpp
    src/hacs/HeckeExt/HeckeCensusSplitValidation.cpp
    src/hacs/HeckeExt/HeckeThetaCensusRingRefusalValidation.cpp
    src/hacs/HeckeExt/HeckeHidaTower.cpp
    src/hacs/HeckeExt/HeckeHidaTowerValidation.cpp
    src/hacs/HeckeExt/HeckeCensusCellList.cpp
    src/hacs/HeckeExt/HeckeCensusCellListValidation.cpp
    src/hacs/HeckeExt/HeckeFormsHeckeAlgebra.cpp
    src/hacs/HeckeExt/HeckeFormsHeckeAlgebraValidation.cpp
    src/hacs/HeckeExt/HeckeFinalityRoutes.cpp
    src/hacs/HeckeExt/HeckeHidaControlValidation.cpp
    src/hacs/HeckeExt/HeckeCensusLattice.cpp
    src/hacs/HeckeExt/HeckeCensusLatticeValidation.cpp
    src/hacs/HeckeExt/HeckeHidaHeavyCellProbe.cpp
    src/hacs/HeckeExt/HeckeMixedFinalityValidation.cpp
    src/hacs/HeckeExt/HeckeBocksteinCuspValidation.cpp
    src/hacs/HeckeExt/HeckeRingHeckeAlgebra.cpp
    src/hacs/HeckeExt/HeckeRingLoewyValidation.cpp
    src/hacs/HeckeExt/HeckeHidaComponents.cpp
    src/hacs/HeckeExt/HeckeHidaComponentsValidation.cpp
    src/hacs/HeckeExt/HeckeSaturationDefect.cpp
    src/hacs/HeckeExt/HeckeSaturationDefectValidation.cpp
    src/hacs/HeckeExt/HeckeHidaCM.cpp
    src/hacs/HeckeExt/HeckeHidaCMValidation.cpp
    src/hacs/HeckeExt/HeckeSmallCarriesValidation.cpp
    src/hacs/HeckeExt/HeckeSatCertDumpValidation.cpp
    src/hacs/HeckeExt/HeckeLatticeModule.cpp
    src/hacs/HeckeExt/HeckeLatticeModuleValidation.cpp
    src/hacs/HeckeExt/LValue/Bernoulli.cpp
    src/hacs/HeckeExt/LValue/LValueRoute.cpp
    src/hacs/HeckeExt/LValue/EisensteinBernoulliRoute.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueValidation.cpp
    src/hacs/HeckeExt/LValue/TwistedEisensteinCell.cpp
    src/hacs/HeckeExt/LValue/BirchModularSymbol.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueBirchArms.cpp
    src/hacs/HeckeExt/LValue/HidaPairing.cpp
    src/hacs/HeckeExt/LValue/HidaPairingRoute.cpp
    src/hacs/HeckeExt/LValue/HidaPairingValidationArms.cpp
    src/hacs/HeckeExt/HeckeHidaLambdaForms.cpp
    src/hacs/HeckeExt/HeckeHidaLambda.cpp
    src/hacs/HeckeExt/HeckeLambdaValidation.cpp
    src/hacs/HeckeExt/LValue/MBlockReader.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueLV3Arms.cpp
    src/hacs/HeckeExt/LValue/ModularSymbolMeasure.cpp
    src/hacs/HeckeExt/LValue/LFunctionRecord.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueMeasureValidation.cpp
    src/hacs/HeckeExt/HeckeLambdaQExp.cpp
    src/hacs/HeckeExt/HeckeLambdaQExpValidation.cpp
    src/hacs/HeckeExt/HeckeLambdaModSym.cpp
    src/hacs/HeckeExt/HeckeLambdaModSymValidation.cpp
    src/hacs/HeckeExt/HeckeLambdaTwoVar.cpp
    src/hacs/HeckeExt/HeckeLambdaTwoVarValidation.cpp
    src/hacs/HeckeExt/LValue/CoefficientRingMeasure.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueCoefficientRingValidation.cpp
    src/hacs/HeckeExt/LValue/EisensteinMeasure.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueEisensteinMeasureValidation.cpp
    src/hacs/HeckeExt/LValue/WildCharacter.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueWildCharacterValidation.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueIwasawaInvariantsValidation.cpp
    src/hacs/HeckeExt/HeckeLambdaTower.cpp
    src/hacs/HeckeExt/HeckeLambdaTowerValidation.cpp
    src/hacs/HeckeExt/HeckeNebentypusWildValidation.cpp
    src/hacs/HeckeExt/LValue/Gamma0NShapiro.cpp
    src/hacs/HeckeExt/LValue/HeckeLValueShapiroValidation.cpp
    src/hacs/HeckeExt/HeckePrecisionReuseValidation.cpp
    src/hacs/HeckeExt/HeckeFormsFastMulValidation.cpp
    src/hacs/HeckeExt/HeckeModuleStreamValidation.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 FNTT3
    # HeckeFormsNtt3Validation (standard): the three-prime engine against independent oracles, the refusal lifted.
    src/hacs/HeckeExt/HeckeFormsNtt3Validation.cpp
    # --- slot 24 CL8
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsSturm.cpp
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsAlgebra.cpp
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsRecords.cpp
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsValues.cpp
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsComplex.cpp
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbsGallery.cpp
)

# Slot 25 (folded at the slot-25 close, S25 FOLD; the former cmake/slot25/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 25 HeckeZLocal
    src/hacs/HeckeExt/HeckeZLocalSource.cpp
    # --- slot 25 HeckeZRecord
    src/hacs/HeckeExt/HeckeZRecordSource.cpp
    src/hacs/HeckeExt/RegisterHeckeZRecordVerbs.cpp
    # --- slot 25 HeckeHidaEisLambda
    src/hacs/HeckeExt/HeckeHidaEisLambda.cpp
    # HeckeHidaEisLambdaValidation (standard)
    src/hacs/HeckeExt/HeckeHidaEisLambdaValidation.cpp
)

# Slot 26 (folded at the slot-26 close, S26 FOLD; the former cmake/slot26/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 26 HeckeHidaDiscLambda (Z6 = H2: the discriminant series of the ramified ordinary families at tame level 1)
    src/hacs/HeckeExt/HeckeHidaDiscLambda.cpp
    src/hacs/HeckeExt/HeckeHidaDiscLambdaValidation.cpp
)
