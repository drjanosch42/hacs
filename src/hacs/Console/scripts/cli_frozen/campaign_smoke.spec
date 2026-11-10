# The CampaignSmokeValidation spec (CampaignValidation.cpp `kSpec`), as a FILE,
# so that `--campaign <spec> --dry-run` is a frozen CLI form that
# ConsoleCliPreservationValidation can byte-compare (console plan v3 Sec. 10
# C1a acceptance (c)).  20 jobs, every one of them also a Golden/ grid point.
# `output` is only a dry-run placeholder: --dry-run writes nothing.
name    = campaign_smoke
model   = sl2z_7vertex
certify = full
lean    = false
output  = /tmp/console_cli_frozen_campaign_smoke
n_sym   = 0,2,4,10,16
ell     = 5,134217757
p       = 2
r       = 1
---
n_sym   = 0,2
ell     = 2
p       = 2,3
r       = 1,2
---
n_sym   = 4
ell     = 134217757
p       = 3
r       = 1,2
