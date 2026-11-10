# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
#
# (R1 / rename; S26 INTEG, R1 audit N5) POST_BUILD alias cohomology_app -> hacs next to the binary.  Non-fatal: on a file
# system without symlinks (Android shared storage, some DrvFs mounts) it warns and the build succeeds.
file(CREATE_LINK hacs "${OC_LINK_DIR}/cohomology_app" RESULT oc_rc SYMBOLIC)
if(NOT oc_rc STREQUAL "0")
    message(WARNING "alias cohomology_app -> hacs not created in ${OC_LINK_DIR}: ${oc_rc} (run the binary as hacs)")
endif()
