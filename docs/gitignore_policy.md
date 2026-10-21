# Git tracking policy

## Keep in Git

- Public copies of the supplied `.py` and `.ipynb` sources, including the ViT training bodies and the ConvNeXtV2/MLP-Mixer launch notebooks.
- README/documentation, source mapping, public configuration, dependency files when verified, and the future CLI/Colab examples.
- The four supplied SSL log/plot files retained for provenance: two metric CSVs and two learning-curve PNGs.

Do not blanket-ignore `*.ipynb`, `*.csv`, `*.json`, `*.png`, or every `log/` directory. Public configurations, source mapping, and the retained original logs use these formats. The recovered `ek_schedule_exe.py` at repository root is tracked as source code.

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

`.gitignore` applies to untracked paths. It does not remove data already in Git history, prevent a force-add, sanitize notebook cell contents or output cells, or inspect metadata inside a checkpoint. Personal path/example redactions are recorded separately; future source additions and generated files still require review. No Git repository has been initialized by this preparation.

Actual output paths are configurable. Use the ignored `outputs/` directory for new examples, and review the full intended file list before committing. If a public synthetic CSV or image is later intentionally included, review it and allow that exact file explicitly.
