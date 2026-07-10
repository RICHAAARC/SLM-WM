# Colab Notebooks

本目录仅保存 Colab 入口。Notebook 不得定义函数、类或直接导入 `main` 与 `experiments` 实现方法。

正式运行顺序:

1. `semantic_watermark_image_only_run.ipynb`。
2. 四个 `external_baseline_*_run.ipynb` 和需要的 `official_reference_*_run.ipynb`。
3. `dataset_level_quality_run.ipynb`。
4. `paper_result_closure_run.ipynb`。

主方法和 baseline 可在独立 Colab 会话并行运行, 结果闭合必须等待全部受治理结果包到达。同一 Notebook 通过 `SLM_WM_PAPER_RUN_NAME` 切换 `probe_paper`、`pilot_paper` 与 `full_paper`, 不允许改变方法机制或实验门禁。
