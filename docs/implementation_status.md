# Implementation status

Updated 2026-10-21; documentation revision 0.1.6.

## Prepared

- Preserved the private baseline of 35 supplied files; retained filenames and notebook formats in the public copy.
- Generalized personal/internal paths and the identifying-looking image example in source text, comments, and notebooks.
- Included the recovered `ek_schedule_exe.py` unchanged at repository root: 36 supplied assets total.
- Recorded source/public SHA-256 hashes, edit status, and reasons without exposing the removed values.
- Updated source verification, Linux/WSL/Colab path/import guidance, and current security findings.
- Rechecked all 71 public files and removed residual personal Colab metadata from two notebooks; notebook cell sources, settings and Python scripts are unchanged.
- Prepared a staged publication plan and checkpoint-selection reporting guidance.
- Recorded the author's confirmed selection rule: review training and validation losses and choose the lowest-validation-loss epoch judged without apparent overfitting.
- Added the MIT License with Copyright (c) 2026 Eisuke Kagawa, as selected by the author. The initial GitHub release v0.1.0 has since been published at commit d6965d35986729e4ccbd402db15730321e42b271.

- Added the unchanged 2025-12-11 snapshot of 165 pinned packages directly as root requirements.txt, reported training platforms, and WSL installation guidance. The snapshot date is kept in documentation; the public requirements filename has no date suffix. The currently reported local driver is distinguished from the historical training environment. Clean installation and reproduction remain pending.

## Release and remaining verification

1. Confirm intended public files and exclude all private originals, backups, patient ECG data, participant records, and runtime outputs.
2. The author confirmed that no third-party source code was copied or incorporated. External library dependencies retain their own terms.
3. Link the confirmed selection criterion to each actual checkpoint, epoch/step, validation split, ordinary/SWA state, and originating run/initialization chain. See `checkpoint_selection.md`; document unresolved correspondence and data-access restrictions honestly.
4. Historical dependency versions and setup guidance are now recorded. Validate installation in a clean WSL/Linux environment and in a fresh Colab runtime; retain the distinction between reported historical settings and newly tested environments.
5. Keep the published v0.1.0 tag unchanged. Review and commit the dependency/documentation additions before creating a subsequent release; they are not present in v0.1.0.

## Later model distribution and convenience features

1. Inspect actual final checkpoints and all configuration/processor metadata; determine a separate weight license.
2. Implement shared inference and compare per-model/ensemble predictions against original results under controlled TTA.
3. Add four CLI examples and a Colab notebook; record verified dependencies and test Linux/WSL/Colab execution.
4. Prepare model assets and checksums for GitHub Releases, then archive the paper-specific release on Zenodo.

The human-study administrator targets an interactive Linux/WSL terminal with a graphical display. No training, inference, participant session, or model reproduction was performed during publication preparation. Static source checks and hashing do not establish scientific reproducibility.
