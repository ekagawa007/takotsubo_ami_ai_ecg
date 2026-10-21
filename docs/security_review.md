# Static security and privacy review

Updated 2026-10-21; documentation revision 0.1.4. [Japanese report](security_review_ja.md).

## Scope

The initial exact-copy snapshot contained 35 supplied files: 24 Python scripts, seven notebooks, two training-metric CSVs, and two learning-curve PNGs. The supplied scheduling helper adds one Python script, for 36 source assets. Original source hashes remain in the mapping; edited public copies have separate hashes and are marked `verified_publication_edit`. The private original files are retained outside this public tree.

## Publication edits completed in this copy

| Finding | Resolution |
|---|---|
| S01: identifying-looking single-image example | Replaced its image filename with `example_ecg.png` and generalized its path. The removed value is not reproduced in public documentation. |
| S02: personal/internal paths | Generalized path roots in Python, notebook cells, comments, and launch examples to documented Linux/Colab locations. Personal workstation labels are replaced consistently with `Local Linux`. Notebook launch examples use the reorganized repository paths. |
| Missing scheduling dependency | Included the supplied `ek_schedule_exe.py` unchanged at repository root and documented the import path. It uses standard-library `datetime` and `time`. |

The changes are recorded in [publication_edits.csv](publication_edits.csv), with original/current hashes in [source_file_mapping.csv](source_file_mapping.csv). Generic paths must be configured before execution. Static source checks do not establish that training, inference, or the reported scientific results have been reproduced.

## Renewed pre-upload review (2026-10-21)

The 71-file public candidate was rechecked against the previous manifest. All 71 hashes matched before this correction. The only nine additional local files were old backups under `.cache/`; these are excluded from publication, including manual browser uploads.

The review found residual personal metadata despite all saved output lists being empty. This revision removes:

- `metadata.colab.authorship_tag` and `metadata.colab.mount_file_id` from the ensemble launcher notebook and the ConvNeXtV2 three-class launcher notebook.
- Cell `metadata.executionInfo`, including executor display name and user ID, from the ensemble launcher notebook.

These identifiers are not reproduced here. Notebook cell sources, ordering, execution counts, outputs, model settings and notebook format are unchanged. Original source hashes are retained; revised public hashes and edit reasons are recorded in the mapping and edit log. The earlier output-only notebook check did not catch these metadata fields.

Additional checks found no common credential signatures (API tokens, passwords or private keys), personal home-directory paths, or residual known workstation/user markers in the public files. All seven notebooks have empty outputs, null execution counts and no attachments. Saved widget state in two notebooks contains progress displays; no embedded ECG image or personal identifier was found there. The two historical CSVs contain checkpoint/training metrics rather than participant records. The two PNGs were visually checked as learning curves; their text metadata identifies Matplotlib only.

Static code inspection found no explicit HTTP/SMTP/FTP upload or suspicious shell/eval/exec operations in the research code. Installing dependencies, retrieving models or metrics, and mounting Google Drive can still communicate externally when notebooks are run. No research code or dependency installation was executed during this review.

This is a review of the supplied files, not a guarantee about future changes, third-party dependencies, runtime behavior, model weights or datasets. No local Git history was present to inspect. Public GitHub contents have not been verified.

## Remaining conditions for execution and later asset sharing

- Dataset evaluators and ensemble prediction scripts write input/checkpoint paths to CSV/JSON. Keep real prediction outputs and input manifests private.
- Visualizations may include the original ECG image and identifying filenames. A Grad-CAM overlay does not anonymize the underlying image.
- The human-study administrator collects participant information and writes responses/timing to CSV. No real participant records are included in the supplied source assets. Keep runtime records private; use synthetic records for demonstrations.
- Checkpoint loading uses `torch.load` in the original research sources. Its behavior depends on the installed PyTorch version and file provenance. Do not load untrusted checkpoints. New inference code should explicitly validate supported loading options and asset hashes.
- Some checkpoints save execution arguments. Inspect actual model assets and metadata before distributing weights; none are included or audited here.
- Original notebook install commands include unpinned upgrades. Dependency/version and model-download requirements still need documentation and testing.
- Human-study free-text CSV fields can be interpreted as spreadsheet formulas, and concurrent writers can conflict. These are conditional runtime risks; the administrator has not been redesigned as a public multiuser application.
- Calling the scheduling helper can sleep until the next day. Configure scheduling and runtime timezone before launching a historical run.

The supplied notebooks have no saved outputs; historical logs/plots are retained unchanged. The earlier static review found no common hardcoded credential signatures in the supplied sources. This is a bounded review, not a guarantee for future files, Git history, runtime outputs, datasets, or checkpoints. The release process must review the complete staged file list and exclude private backups and originals. `.gitignore` does not remove tracked history or sanitize file contents.

## References

- [GitHub: removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [PyTorch: torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch: serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html)
