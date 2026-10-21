# takotsubo_ami_ai_ecg

**Documentation version:** `0.1.4`  
**Last updated:** 2026-10-21 (JST)

Research code for ECG-image models distinguishing takotsubo syndrome from acute anterior STEMI, including model development, ensemble evaluation, and a human diagnostic study.

## Current status

The research-source snapshot contains the 35 originally supplied files and the recovered `ek_schedule_exe.py` scheduling helper. Original filenames and notebook formats are retained. Public copies generalize personal paths, workstation labels, and the identifying-looking image example; original and current SHA-256 hashes and edit reasons are recorded in the [source mapping](docs/source_file_mapping.csv) and [publication edit log](docs/publication_edits.csv). Personal Colab execution metadata has also been removed from two notebooks; cell sources and settings are unchanged. The helper is supplied unchanged. This snapshot is prepared for a staged source-code release; final-run identification, dependency/environment documentation, and model execution validation remain pending. The code license is MIT, with copyright held by Eisuke Kagawa. See the [publication plan (Japanese)](docs/publication_plan_ja.md) and [security review](docs/security_review.md).

## Contents

| Location | Purpose |
|---|---|
| `ek_schedule_exe.py` | Original helper for waiting until a specified local time |
| `training/pretrain/` | Original SSL code and supplied historical logs |
| `training/ft_3_class/` | Original three-class fine-tuning code and notebooks |
| `training/ft_2_class/` | Original two-class fine-tuning code and notebooks |
| `evaluation/original/` | Original single-image inference, dataset evaluation, and ensemble scripts |
| `human_test/` | Original study preparation, randomization, and administration scripts |
| `src/takotsubo_ami_ai_ecg/` | Reserved for shared public inference implementation |
| `examples/` | Reserved for four CLI examples and a Colab notebook |
| `configs/` | Reserved for verified inference configuration and weight manifest |
| `tests/` | Reserved for prediction-parity and loading checks |
| `weights/` | Local download destination; weights are not committed |
| `outputs/` | Local runtime outputs; generated files are not committed |

## Development provenance

ViT fine-tuning was developed interactively in Jupyter/Google Colab, and its original training implementation is therefore provided in `.ipynb` format. These notebooks are the original training code, not only demonstrations. During the subsequent development of ConvNeXtV2 and MLP-Mixer, the training implementations were organized into `.py` scripts for convenience; their launcher notebooks are also retained because they record run arguments. Original filenames, notebook formats, and learning procedures are preserved; publication edits to paths and examples are recorded separately. The shared public inference package and CLI/Colab examples will be developed separately.

- [Development code and execution context](docs/development_code.md)
- [Source-to-destination mapping](docs/source_file_mapping.md)
- [Full mapping and SHA-256 hashes](docs/source_file_mapping.csv)
- [Recording the selected model checkpoints](docs/checkpoint_selection.md)
- [Remaining implementation work](docs/implementation_status.md)
- [Static security and privacy review](docs/security_review.md)

## Checkpoint selection

Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected.

The selected checkpoint filename, epoch/global step, originating run, and ordinary versus averaged weight state still need to be linked to the actual evaluation records. See [checkpoint selection details](docs/checkpoint_selection.md).

## Execution environments

The execution targets are Linux, including WSL on Windows 11 and Google Colab. Windows 11 can host and edit the repository; its native Python interpreter is not a target for running the research models or public inference examples.

| Workflow | Target environment | Status |
|---|---|---|
| Shared inference and four CLI examples | Linux/WSL; the same Python implementation in Colab | To be implemented and validated |
| Original training and evaluation | Linux/WSL or Colab, with original dependencies and paths configured | Source retained; reproduction pending |
| Human-study administrator | Interactive Linux/WSL terminal with working graphical display | Source retained; environment validation pending |

The original human-study administrator uses terminal keyboard input and graphical windows, so it is not a standard Colab notebook workflow. See [Linux and Colab environment plan](docs/execution_environment.md). GPU execution and dependency versions will be documented after testing with the actual model checkpoints.

## Verify source integrity

Run from the repository root in Linux/WSL using Python 3.10 or later:

```console
python3 tools/verify_source_mapping.py
```

This compares the current public-source files with their recorded destination hashes. Unedited files must also match the original hashes; edited files are labeled `verified_publication_edit`. It does not import, execute, or train any research model. An optional `--source-root` argument checks the private original source directory when available, including the recovered helper.

## Planned distribution

The proposed first public release contains the research training, original evaluation, and human-study administration source with documentation and the scheduling helper. Patient ECG data, participant records, credentials, private originals, and local backups are excluded. Convenient inference examples and a new Colab notebook will follow. Code and documentation are intended for GitHub. The final three models and required loading configuration will be distributed through GitHub Releases. The paper-specific release will later be archived on Zenodo for a version-specific DOI.

There are currently no runnable public quick-start examples, published weight download URLs, dependency lock files, or validated Linux/WSL/Colab installation instructions in this snapshot. No placeholder URLs or unverified dependency versions are supplied.

## License and citation

The code is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Eisuke Kagawa. Third-party components retain their respective licenses and notices. Model weights are not included in this code release; their license will be documented separately after reviewing the actual assets and upstream terms. Author list, paper metadata, GitHub owner, and release identifiers remain to be confirmed. See the [publication and licensing plan](docs/publication_plan_ja.md). Documentation version `0.1.4` identifies this README revision; the proposed first repository release tag is `v0.1.0` and has not been created.

## Revision history

Documentation versions track README revisions during repository preparation. For each future revision, update the version and date above and add an entry below, with the newest entry first.

| Date (JST) | Documentation version | Changes |
|---|---|---|
| 2026-10-21 | 0.1.4 | Removed personal Colab authorship, Drive-file and executor metadata from two notebooks after a renewed 71-file security review; updated public hashes and review records. Notebook cell sources, model settings and Python scripts are unchanged. |
| 2026-10-21 | 0.1.3 | Recorded the author-confirmed checkpoint-selection method: review training and validation losses and select the lowest-validation-loss epoch without apparent signs of overfitting. Research sources are unchanged; exact checkpoint identities remain to be documented. |
| 2026-10-21 | 0.1.2 | Added the MIT License with Copyright (c) 2026 Eisuke Kagawa; recorded the confirmed licensing decision and checkpoint-selection reporting guidance. Research sources and model settings are unchanged. |
| 2026-10-21 | 0.1.1 | Generalized personal/internal paths, workstation labels, and the identifying-looking example filename; included the supplied scheduling helper unchanged; recorded original/public hashes and edit reasons for 36 supplied assets; updated verification and execution guidance; added a staged publication and licensing plan. Model execution validation remains pending. |
| 2026-10-21 | 0.1.0 | Introduced documentation versioning, the last-updated date, and this revision history. Recorded the current preparation state: 35 original source files preserved with SHA-256 verification; ViT fine-tuning retained as original notebooks and ConvNeXtV2/MLP-Mixer training retained as scripts with launcher notebooks; Linux/WSL/Colab execution targets documented, with interactive Linux/WSL required for the human-study administrator; ignore rules expanded for local data, checkpoints, study records, and generated outputs. Model execution validation remains pending. |
