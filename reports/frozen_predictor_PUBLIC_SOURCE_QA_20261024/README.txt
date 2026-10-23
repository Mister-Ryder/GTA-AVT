Released frozen-classifier source and actual public-mapped CUDA QA

The nine new source/configuration/export files define an isolated 39-file snapshot. Use docs/reproduce_frozen_ordinal_source.txt and tools/export_frozen_ordinal_source.py to reproduce the source identity.

The exact exported public snapshot passed independent actual CUDA checks on two devices: synthetic 128-cubed inputs, batch two, AMP, and one disposable successful update per C/D arm. Full classifier parameters and buffers stayed fixed, evaluation outputs repeated exactly, initialization and zero-interaction C/D parity held, and intended trainable routes updated. Source and receipt bytes were verified before and after transfer.

The source identity file was frozen before these public checks and keeps its original public_mapped_GPU_QA_executed=false field. engineering_QA.json records the later actual public-mapped execution; the historical identity was not rewritten. Private scientific-source QA does not certify a different public export.

This is an engineering check using synthetic inputs. It does not test segmentation accuracy, anatomical correspondence, independent cold predictor pretraining, a new scientific block, or generalization. The private scientific and public-mapped snapshots have explicitly documented source mappings; they are not byte-identical. No study images, geometry arrays, trained weights or device identifiers are released here.
