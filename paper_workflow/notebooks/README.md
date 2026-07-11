# Colab Notebooks

本目录仅保存 Colab 入口。Notebook 不得定义函数、类或直接导入 `main` 与 `experiments` 实现方法。

正式运行顺序:

1. `semantic_watermark_image_only_run.ipynb`。
2. 三个 `external_baseline_*_run.ipynb`、`official_reference_t2smark_run.ipynb` 和补充方法忠实度所需的其他 `official_reference_*_run.ipynb`。
3. `paper_result_closure_run.ipynb`。

主方法入口在完成全部 Prompt 后释放生成模型显存, 随即从真实 clean / watermarked
图像对提取正式 Inception 特征并计算 FID / KID。因此数据集质量不是独立 Notebook
协议, 不存在第二套结果导入或质量计算路径。

主方法和 baseline 可在独立 Colab 会话并行运行, 结果闭合必须等待全部受治理结果包到达。同一 Notebook 通过 `SLM_WM_PAPER_RUN_NAME` 切换 `probe_paper`、`pilot_paper` 与 `full_paper`, 不允许改变方法机制或实验门禁。

Notebook 的状态展示路径必须从 `SLM_WM_PAPER_RUN_NAME` 构造, 不读取 artifact 全局目录。Notebook runtime 报告独立写入 `outputs/notebook_runtime_observation/<paper_run_name>/...`, 不进入10类正式 GPU 输入包。正式运行未通过 ready 门禁时保留诊断文件, 但不会生成可供 CPU 闭合选择的 ZIP。
