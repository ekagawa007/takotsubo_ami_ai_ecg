# Source file mapping

Updated 2026-10-21; documentation revision 0.1.4.

The initial 35-file layout was verified byte-for-byte. This public preparation snapshot contains those 35 source-derived files plus the supplied scheduling helper (36 assets). Personal paths and example identifiers are generalized in edited files. Personal Colab execution metadata was removed from two notebooks in revision 0.1.4 without changing cell sources or settings. Original source hashes are retained, and public hashes and edit status are recorded separately. Hashes verify file integrity; they do not establish which run or checkpoint produced the paper's results.

Paths are relative to the private original source root and the public repository root. The helper was supplied at the original source root. See [publication_edits.csv](publication_edits.csv) for reasons and static checks; removed private values are not included in this record.

| Source path | Destination path | Role | Verification status |
|---|---|---|---|
| `ft_2_class/ConvNeXtV2/evaluate_dataset_dir_convnextv2.py` | `evaluation/original/convnextv2/evaluate_dataset_dir_convnextv2.py` | original_inference_evaluation | verified_identical |
| `ft_2_class/ConvNeXtV2/predict_single_image_convnextv2.py` | `evaluation/original/convnextv2/predict_single_image_convnextv2.py` | original_inference_evaluation | verified_identical |
| `ft_2_class/ConvNeXtV2/tako_ami_ft2_convnextv2_20251004_2.py` | `training/ft_2_class/convnextv2/tako_ami_ft2_convnextv2_20251004_2.py` | ft2_source | verified_publication_edit |
| `ft_2_class/ConvNeXtV2/tako_ami_ft2_convnextv2_gd_20251010m.ipynb` | `training/ft_2_class/convnextv2/tako_ami_ft2_convnextv2_gd_20251010m.ipynb` | ft2_notebook | verified_publication_edit |
| `ft_2_class/ensemble/predict_ensemble3_testset_from_csv_rawlogits.py` | `evaluation/original/ensemble/predict_ensemble3_testset_from_csv_rawlogits.py` | original_ensemble | verified_publication_edit |
| `ft_2_class/ensemble/predict_single_image_ensemble.py` | `evaluation/original/ensemble/predict_single_image_ensemble.py` | original_ensemble | verified_identical |
| `ft_2_class/ensemble/tako_ami_ensemble3_gd_from_csv_20251210_4m.ipynb` | `evaluation/original/ensemble/tako_ami_ensemble3_gd_from_csv_20251210_4m.ipynb` | original_ensemble | verified_publication_edit |
| `ft_2_class/MLP-Mixer/evaluate_dataset_dir_mlpmixer.py` | `evaluation/original/mlpmixer/evaluate_dataset_dir_mlpmixer.py` | original_inference_evaluation | verified_identical |
| `ft_2_class/MLP-Mixer/predict_single_image_mlpmixer.py` | `evaluation/original/mlpmixer/predict_single_image_mlpmixer.py` | original_inference_evaluation | verified_publication_edit |
| `ft_2_class/MLP-Mixer/tako_ami_ft2_mlpmixer_20251102_3.py` | `training/ft_2_class/mlpmixer/tako_ami_ft2_mlpmixer_20251102_3.py` | ft2_source | verified_identical |
| `ft_2_class/MLP-Mixer/tako_ami_ft2_mlpmixer_gd_20251110_3m.ipynb` | `training/ft_2_class/mlpmixer/tako_ami_ft2_mlpmixer_gd_20251110_3m.ipynb` | ft2_notebook | verified_publication_edit |
| `ft_2_class/ViT/evaluate_dataset_dir_ViT.py` | `evaluation/original/vit/evaluate_dataset_dir_ViT.py` | original_inference_evaluation | verified_identical |
| `ft_2_class/ViT/predict_single_image.py` | `evaluation/original/vit/predict_single_image.py` | original_inference_evaluation | verified_publication_edit |
| `ft_2_class/ViT/tako_ami_ft2_vit_prod_gd_20250817_5m.ipynb` | `training/ft_2_class/vit/tako_ami_ft2_vit_prod_gd_20250817_5m.ipynb` | ft2_notebook | verified_publication_edit |
| `ft_3_class/ConvNeXtV2/tako_ami_ft3_convnextv2_gd_20251003_4.py` | `training/ft_3_class/convnextv2/tako_ami_ft3_convnextv2_gd_20251003_4.py` | ft3_source | verified_publication_edit |
| `ft_3_class/ConvNeXtV2/tako_ami_ft3_convnextv2_gd_20251009m.ipynb` | `training/ft_3_class/convnextv2/tako_ami_ft3_convnextv2_gd_20251009m.ipynb` | ft3_notebook | verified_publication_edit |
| `ft_3_class/MLP-Mixer/tako_ami_ft3_mlpmixer_20251014_3.py` | `training/ft_3_class/mlpmixer/tako_ami_ft3_mlpmixer_20251014_3.py` | ft3_source | verified_identical |
| `ft_3_class/MLP-Mixer/tako_ami_ft3_mlpmixer_gd_20251019m.ipynb` | `training/ft_3_class/mlpmixer/tako_ami_ft3_mlpmixer_gd_20251019m.ipynb` | ft3_notebook | verified_publication_edit |
| `ft_3_class/ViT/tako_ami_ft3_vit_prod_gd_20250814_4m.ipynb` | `training/ft_3_class/vit/tako_ami_ft3_vit_prod_gd_20250814_4m.ipynb` | ft3_notebook | verified_publication_edit |
| `human_test/copy_img_to_dataset_dir.py` | `human_test/copy_img_to_dataset_dir.py` | human_study_source | verified_identical |
| `human_test/generate_stratified_block_randomization_table.py` | `human_test/generate_stratified_block_randomization_table.py` | human_study_source | verified_publication_edit |
| `human_test/make_human_test_file_list.py` | `human_test/make_human_test_file_list.py` | human_study_source | verified_publication_edit |
| `human_test/tako_ami_human_test_with_random.py` | `human_test/tako_ami_human_test_with_random.py` | human_study_source | verified_publication_edit |
| `pretrain/ConvNeXtV2/evaluate_fcmae.py` | `training/pretrain/convnextv2/evaluate_fcmae.py` | ssl_source | verified_identical |
| `pretrain/ConvNeXtV2/log/evaluation_results.csv` | `training/pretrain/convnextv2/log/evaluation_results.csv` | historical_training_log | verified_identical |
| `pretrain/ConvNeXtV2/lr_finder_convnextv2_fcmae.py` | `training/pretrain/convnextv2/lr_finder_convnextv2_fcmae.py` | ssl_source | verified_identical |
| `pretrain/ConvNeXtV2/mae_fcmae_wrapper.py` | `training/pretrain/convnextv2/mae_fcmae_wrapper.py` | ssl_source | verified_identical |
| `pretrain/ConvNeXtV2/train_convnextv2_fcmae.py` | `training/pretrain/convnextv2/train_convnextv2_fcmae.py` | ssl_source | verified_identical |
| `pretrain/MLP-Mixer/evaluate_mlpmixer_mae.py` | `training/pretrain/mlpmixer/evaluate_mlpmixer_mae.py` | ssl_source | verified_identical |
| `pretrain/MLP-Mixer/log/evaluation.csv` | `training/pretrain/mlpmixer/log/evaluation.csv` | historical_training_log | verified_identical |
| `pretrain/MLP-Mixer/log/lr_find_plot.png` | `training/pretrain/mlpmixer/log/lr_find_plot.png` | historical_training_log | verified_identical |
| `pretrain/MLP-Mixer/train_mlpmixer_mae.py` | `training/pretrain/mlpmixer/train_mlpmixer_mae.py` | ssl_source | verified_identical |
| `pretrain/ViT/log/eval_loss_curve.png` | `training/pretrain/vit/log/eval_loss_curve.png` | historical_training_log | verified_identical |
| `pretrain/ViT/tako_ami_pret_vit_prod_20250722.py` | `training/pretrain/vit/tako_ami_pret_vit_prod_20250722.py` | ssl_source | verified_publication_edit |
| `pretrain/ViT/tako_ami_pretrain_vit_mae_eval_loss.py` | `training/pretrain/vit/tako_ami_pretrain_vit_mae_eval_loss.py` | ssl_source | verified_publication_edit |
| `ek_schedule_exe.py` | `ek_schedule_exe.py` | original_scheduling_helper | verified_identical |

Full original/current SHA-256 values are recorded in [source_file_mapping.csv](source_file_mapping.csv). `verified_identical` requires equality to the original; `verified_publication_edit` requires equality to the recorded public version and refers to the edit log.

Keep the original files private. Do not overwrite source hashes to hide an edit. For integrity verification, run `python3 tools/verify_source_mapping.py`; optionally supply `--source-root` for the private original directory.
