# Original human-study scripts

The four supplied scripts retain their filenames and processing logic; personal paths and examples are generalized in the public copy:

- `copy_img_to_dataset_dir.py`: prepare/copy question images.
- `make_human_test_file_list.py`: prepare question manifests.
- `generate_stratified_block_randomization_table.py`: generate allocation schedules.
- `tako_ami_human_test_with_random.py`: administer the study and save responses.

Generic paths, study-specific settings and dependencies must be checked before execution. The administrator targets an interactive Linux/WSL terminal with a working graphical display. It uses POSIX-specific modules, terminal keyboard input and graphical windows. A standard Google Colab notebook does not provide the same interaction model; a browser-based human-study interface would be separate work. Native Windows is not a target runtime. No study session was run during preparation. See [execution environments](../docs/execution_environment.md).

Participant details, responses, allocation state and ECG/question image files are not part of this source release. The administrator collects participant information at runtime; generated records belong in private storage. See [security review](../docs/security_review.md) for the static findings and proposed public-demo changes.
