# Historical source provenance

The three Python operator sources in this directory are copied byte for byte
from the earlier GTAirway implementation. The v1.1 wrapper assembles only their
active encoder, decoder, fusion and attention primitives. It does not instantiate
the obsolete global attribute heads or the unused historical segmentation heads.

These sources descend in part from NaviAirway by Anton Wang (2022). Its original
MIT notice is preserved in `NaviAirway-MIT.txt`. This notice is not a blanket MIT
license for later GTAirway research additions or this complete project; no
independent license for those additions has been selected here.

| File | SHA256 of unmodified historical source |
| --- | --- |
| model_arch_model10.py | 55d51ea850efd2b31aa07acd68ad5e24f64fafe6a714d26dc357919f639f3003 |
| AnatomicalClassifier.py | f5288c74ce71db97eca63f9665e29dfb0099aa010326ce7d6748377bc0156222 |
| AnatomicalFusionModules.py | 7309281bb296eb7d861050f5ead6b563823b8393addf6ff7c18a594225030b01 |
