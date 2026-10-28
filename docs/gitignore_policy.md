# Git tracking policy

Updated 2026-10-28 (JST); documentation revision 0.1.21.

## Keep in Git

- Public copies of the supplied `.py` and `.ipynb` sources, including the ViT training bodies and the ConvNeXtV2/MLP-Mixer launch notebooks.
- README/documentation, CITATION.cff, public configuration, verified inference dependency files, the historical training dependency record, and the CLI/Colab examples.
- The four supplied SSL log/plot files retained for provenance: two metric CSVs and two learning-curve PNGs.

Do not blanket-ignore `*.ipynb`, `*.csv`, `*.json`, `*.png`, or every `log/` directory. Public configurations, notebooks and the retained aggregate training logs use these formats. Internal source mappings and publication/security-review records are kept outside the public tree. The recovered `ek_schedule_exe.py` at repository root is tracked as source code.

## Exclude from Git

| Category | Examples |
|---|---|
| Local environments and caches | `.venv/`, `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.cache/` |
| Secrets and local settings | `.env`, credentials, private keys; safe `.env.example` may be tracked |
| Weight/checkpoint assets | `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, checkpoint directories |
| ECG data and generated outputs | data/dataset directories, `outputs/`, runtime `result/` and `results/` |
| Study records and question material | response CSVs, selected-image manifests, allocation state, question/image directories |
| Runtime logs and temporary files | TensorBoard event files, W&B state, temporary CSVs and `.log` files |

Known study filenames are excluded regardless of working directory because some historical scripts write relative to the current working directory. Local `weights/README.md` and `outputs/README.md` remain trackable so those folders can explain their purpose without committing generated content.

## Limits

`.gitignore` applies to untracked paths. It does not remove data already in Git history, prevent a force-add, sanitize notebook cell contents or output cells, or inspect metadata inside a checkpoint. Personal path/example redactions are recorded separately; future source additions and generated files still require review. The public Git repository and release tags are established; removed files can remain in earlier commits, so reviewing the current tree alone does not inspect all historical content.

Actual output paths are configurable. Use the ignored `outputs/` directory for new examples, and review the full intended file list before committing. If a public synthetic CSV or image is later intentionally included, review it and allow that exact file explicitly.
