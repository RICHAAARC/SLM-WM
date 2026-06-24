# 闃舵鐘舵€?

## 鏂囨。瀹氫綅

鏈枃妗ｈ褰曞綋鍓嶅垎闃舵鏋勫缓鎺ㄨ繘鐘舵€併€傚畠鍙弿杩伴樁娈甸棬绂併€佽緭鍏ャ€佽緭鍑哄拰闃绘柇椤?
涓嶆壙杞芥寮忚鏂囧疄楠岀粨璁恒€?

## stage_00_core_package_boundary_freeze

| item | value |
| --- | --- |
| construction_unit_name | `stage_00_core_package_boundary_freeze` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/audit_reports/harness_audit_summary.json` |
| expected_output_manifest | `outputs/core_package_boundary_freeze/manifest.local.json` |
| expected_outputs | `outputs/core_package_boundary_freeze/core_boundary_report.json`; `outputs/core_package_boundary_freeze/core_import_report.json`; `outputs/core_package_boundary_freeze/core_package_layout.txt`; `outputs/core_package_boundary_freeze/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻ユ牳蹇冨寘杈圭晫妫€鏌ュけ璐? 鍋滄鎺ㄨ繘骞朵慨澶?`main/` 鍙嶅悜渚濊禆銆?|
| invariants | `main/` 涓嶄緷璧?Colab銆丏rive銆乪xperiments銆乻cripts銆乼ests銆乼ools/harness銆乸aper_workflow 鎴栧閮?baseline銆?|
| next_stage_entry | stage00 楠岃瘉閫氳繃鍚? 鎵嶈兘杩涘叆 `stage_01_algorithm_primitives`銆?|

### stage00 宸插喕缁撳唴瀹?

1. `main/` 鏈€灏忓寘缁撴瀯鍖呮嫭 `main/core/`銆乣main/methods/`銆乣main/protocol/`銆乣main/analysis/` 鍜?`main/cli/`銆?
2. `main/core/method_objects.py` 瀹氫箟璇箟鏉′欢銆佹綔绌洪棿瀛愮┖闂淬€佹按鍗拌浇浣撱€佹敞鎰忓姏閿氱偣銆佹娴嬭瘉鎹拰铻嶅悎鍐崇瓥鐨勬渶灏?typed object銆?
3. `tests/constraints/test_main_boundary_contract.py` 瀵规牳蹇冨寘瀵煎叆杈圭晫杩涜杞婚噺绾︽潫娴嬭瘯銆?
4. `scripts/write_core_package_boundary_outputs.py` 鍙悜 `outputs/core_package_boundary_freeze/` 鍐欏叆鏈湴闃舵鎶ュ憡銆?

### stage00 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python -c "import main"` | pass |
| `python scripts/write_core_package_boundary_outputs.py` | pass |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 12 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_01_algorithm_primitives

| item | value |
| --- | --- |
| construction_unit_name | `stage_01_algorithm_primitives` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json` |
| expected_output_manifest | `outputs/algorithm_primitives/manifest.local.json` |
| expected_outputs | `outputs/algorithm_primitives/core_primitive_summary.json`; `outputs/algorithm_primitives/synthetic_core_records.jsonl`; `outputs/algorithm_primitives/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻ョ函绠楁硶鍘熻涓嶈兘鍦ㄦ棤 SD3銆佹棤 Colab銆佹棤 Drive 鐜涓嬮€氳繃娴嬭瘯, 鍋滄鎺ㄨ繘骞朵慨澶?`main/methods/` 鍘熻瀹炵幇銆?|
| invariants | 涓嶅紩鍏?diffusers銆乼ransformers銆丼D 鏉冮噸銆丆olab銆丏rive 鎴?Notebook; `main/` 涓嶅啓鍑?records; attention carrier 浠呬负 synthetic stub銆?|
| next_stage_entry | stage01 楠岃瘉閫氳繃鍚? 鎵嶈兘杩涘叆 `stage_02_core_method_smoke_test`銆?|

### stage01 宸插畬鎴愬唴瀹?

1. `main/methods/algorithm_primitives.py` 瀹炵幇绾畻娉曞師璇棴鐜? 鍖呮嫭璇箟椋庨櫓鍦恒€乴atent mask 鎶曞奖銆佸畨鍏ㄥ熀搴曚及璁°€丩F/HF carrier銆乤ttention synthetic stub銆乴atent update 鍚堟垚銆佸唴瀹瑰垎鏁般€佸嚑浣曞彲闈犳€у拰 evidence/final 鍒ゅ畾銆?
2. `scripts/run_core_smoke.py` 鏍规嵁 typed objects 鐢熸垚 stage01 鏈湴 summary銆乻ynthetic records 鍜?manifest, 涓旀墍鏈夎緭鍑哄潎鍐欏叆 `outputs/algorithm_primitives/`銆?
3. `tests/functional/test_algorithm_primitives.py` 瑕嗙洊姝ｇ‘ key銆侀敊璇?key銆丠F tail truncation銆乺escue 杈圭晫鍜?attestation 鍒嗗眰銆?
4. `docs/field_registry.md` 宸茬櫥璁?stage01 鏂板瀛楁銆?

### stage01 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python scripts/run_core_smoke.py` | pass |
| `pytest tests/functional -q` | pass, 7 passed |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 16 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_02_core_method_smoke_test

| item | value |
| --- | --- |
| construction_unit_name | `stage_02_core_method_smoke_test` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json`; `outputs/algorithm_primitives/manifest.local.json` |
| expected_output_manifest | `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_outputs | `outputs/core_method_synthetic_smoke/synthetic_event_records.jsonl`; `outputs/core_method_synthetic_smoke/core_smoke_metrics.json`; `outputs/core_method_synthetic_smoke/core_smoke_summary.md`; `outputs/core_method_synthetic_smoke/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻?synthetic latent smoke 涓嶈兘澶嶇幇 key 鍖哄垎銆乺escue 杈圭晫鎴?attestation 鍒嗗眰, 鍋滄鎺ㄨ繘骞朵慨澶?`main/methods/synthetic_smoke.py`銆?|
| invariants | 涓嶆帴鍏ョ湡瀹?SD3/SD3.5銆丆olab銆丏rive 鎴?Notebook; 涓嶆妸 smoke 缁撴灉鍐欐垚璁烘枃 supported claims; attention carrier 浠嶄负 synthetic stub銆?|
| next_stage_entry | stage02 楠岃瘉閫氳繃鍚? 鎵嶈兘杩涘叆 `stage_03_sd3_runtime_adapter`銆?|

### stage02 宸插畬鎴愬唴瀹?

1. `main/methods/synthetic_smoke.py` 鏋勯€?clean銆亀atermarked銆亀rong-key negative銆乬eometric shifted銆乤ligned recovered銆乽nattested positive 鍜?final positive 绛?synthetic latent 鍦烘櫙銆?
2. `scripts/run_core_smoke.py --unit core_method_smoke` 鍐欏嚭 stage02 synthetic records銆乵etrics銆乻ummary 鍜?manifest銆?
3. `scripts/run_minimal_method_smoke.py` 鎻愪緵 minimal method package 鍙鐢ㄧ殑 stdout smoke銆?
4. `tests/functional/test_core_method_smoke.py` 瑕嗙洊閿欒 key銆乺escue 杈圭晫銆佸嚑浣曞彲闈犳€т笉瓒抽樆鏂?rescue 鍜?attestation 鍒嗗眰銆?

### stage02 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python scripts/run_minimal_method_smoke.py` | pass |
| `python scripts/run_core_smoke.py --unit core_method_smoke` | pass |
| `pytest tests/functional -q` | pass, 11 passed |
| `pytest tests/constraints -q` | pass, 9 passed |
| `pytest -q` | pass, 20 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_03_sd3_runtime_adapter

| item | value |
| --- | --- |
| construction_unit_name | `stage_03_sd3_runtime_adapter` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_output_manifest | `outputs/sd_runtime_adapter/manifest.local.json` |
| expected_outputs | `outputs/sd_runtime_adapter/sd_generation_records.jsonl`; `outputs/sd_runtime_adapter/latent_trace_records.jsonl`; `outputs/sd_runtime_adapter/attention_capture_records.jsonl`; `outputs/sd_runtime_adapter/generation_quality_summary.json`; `outputs/sd_runtime_adapter/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鏈湴娌℃湁鐪熷疄 SD3 / SD3.5 鏉冮噸銆丟PU 鎴栨ā鍨嬭闂潈闄愭椂, 浣跨敤 synthetic fallback 鐢熸垚宸ョ▼ records, 骞跺湪 records 涓啓鍏?`unsupported_reason`; fallback records 涓嶆敮鎸佹寮忚鏂?claim銆?|
| invariants | `main/` 涓嶄緷璧?diffusers銆乼ransformers銆佹ā鍨嬫潈閲嶃€乪xperiments runtime 鎴栬剼鏈? runtime 灞傚彧鑳借皟鐢?core, core 涓嶅弽鍚戜緷璧?runtime銆?|
| next_stage_entry | stage03 楠岃瘉閫氳繃鍚? 鎵嶈兘杩涘叆 `stage_04_minimal_diffusion_latent_injection`銆?|

### stage03 宸插畬鎴愬唴瀹?

1. `experiments/runtime/diffusion/` 鎻愪緵 SD3 / SD3.5 runtime adapter銆乻ynthetic fallback銆乻ampler hook銆乴atent trace銆乤ttention capture 鍜?latent estimator銆?
2. `configs/model_sd3.yaml` 涓?`configs/model_sd35.yaml` 鎻愪緵杞婚噺 runtime probe 閰嶇疆銆?
3. `scripts/run_diffusion_runtime_probe.py` 鍐欏嚭 generation records銆乴atent trace records銆乤ttention capture records銆乹uality summary 鍜?manifest, 涓旀墍鏈夎緭鍑哄潎鍐欏叆 `outputs/sd_runtime_adapter/`銆?
4. `tests/functional/test_diffusion_runtime_adapter.py` 瑕嗙洊 fallback 鍘熷洜銆佺浉鍚?prompt / seed 澶嶇幇鍜岃緭鍑虹洰褰曠害鏉熴€?
5. `docs/field_registry.md` 宸茬櫥璁?runtime adapter 鏂板瀛楁銆?

### stage03 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/run_diffusion_runtime_probe.py` | pass |
| `pytest tests/functional/test_diffusion_runtime_adapter.py -q` | pass, 3 passed |
| `pytest -q` | pass, 24 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_04_minimal_diffusion_latent_injection

| item | value |
| --- | --- |
| construction_unit_name | `stage_04_minimal_diffusion_latent_injection` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/core_method_synthetic_smoke/manifest.local.json` |
| expected_output_manifest | `outputs/minimal_diffusion_latent_injection/manifest.local.json` |
| expected_outputs | `paper_workflow/sd_runtime_cold_start_probe.ipynb`; `paper_workflow/colab_utils/sd_runtime_cold_start.py`; `paper_workflow/minimal_latent_injection_run.ipynb`; `paper_workflow/colab_utils/minimal_latent_injection.py`; `outputs/real_sd_runtime_probe/*_manifest.local.json`; `outputs/real_sd_runtime_probe/*_environment_report.json`; `outputs/real_sd_runtime_probe_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/real_sd_runtime_probe/real_sd_runtime_probe_package_<utc>_<short_commit>.zip`; `outputs/minimal_diffusion_latent_injection/*_injection_result.json`; `outputs/minimal_diffusion_latent_injection/*_latent_update_records.jsonl`; `outputs/minimal_diffusion_latent_injection/*_paired_quality_metrics.csv`; `outputs/minimal_diffusion_latent_injection/*_environment_report.json`; `outputs/minimal_diffusion_latent_injection/*_manifest.local.json`; `outputs/minimal_latent_injection_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/minimal_diffusion_latent_injection/minimal_latent_injection_package_<utc>_<short_commit>.zip` |
| blocking_items | 鏃犮€?|
| fallback_path | SD3.5 Medium 鏄富绾? 鑻ヤ富妯″瀷鍦?Colab 涓嶅彲鐢? 杩愯 SD3 Medium 鍏煎 fallback 骞跺啓鍑?`unsupported_reason`; fallback 浜х墿涓嶅緱鏀寔姝ｅ紡璁烘枃 claim銆?|
| invariants | Notebook 鍙綔涓哄叆鍙? runtime 閫昏緫浣嶄簬 repository helper; `main/` 涓嶄緷璧?Colab銆丏rive銆乨iffusers銆乼ransformers 鎴栨ā鍨嬫潈閲嶃€?|
| next_stage_entry | Colab 鐪熷疄鎺ㄧ悊銆佺湡瀹?latent trajectory銆乸aired images銆乴atent update records 鍜岃川閲忔寚鏍囧潎宸查€氳繃鏈湴瀹¤; 鍙繘鍏?`stage_05_colab_drive_workflow`銆?|

### stage04 宸插畬鎴愬唴瀹?

1. `paper_workflow/sd_runtime_cold_start_probe.ipynb` 鎻愪緵 Colab 鍐峰惎鍔ㄥ叆鍙? 鏀寔鎷夊彇浠ｇ爜銆佸畨瑁呬緷璧栥€佺櫥褰?Hugging Face銆佹寕杞?Google Drive, 骞跺彲杩愯 SD3.5 Medium 涓绘ā鍨嬩笌 SD3 Medium 鍏煎 fallback銆?
2. `paper_workflow/colab_utils/sd_runtime_cold_start.py` 鎵胯浇鐪熷疄 SD runtime 璋冪敤銆乴atent callback 鎹曡幏銆佸浘鍍忔憳瑕併€乼rajectory records銆乪nvironment report銆乻ummary銆乵anifest銆亃ip 鎵撳寘鍜?Google Drive 闀滃儚閫昏緫銆?
3. 宸插璁?`outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; 璇ュ寘瀵瑰簲鎻愪氦 `b2be25c`, ZIP 瀹屾暣鎬ч€氳繃, SHA-256 涓?`be6e4373edf81311209e0eb220ac189fd43e046128e2ba05815a0775dd9fceb7`銆?
4. runtime probe 缁撴灉涓? SD3.5 Medium 涓绘ā鍨?`stabilityai/stable-diffusion-3.5-medium` 涓?SD3 Medium fallback `stabilityai/stable-diffusion-3-medium-diffusers` 鍧囧畬鎴愮湡瀹炴帹鐞? 鍧囨崟鑾?28 鏉＄湡瀹?latent trajectory records, latent shape 鍧囦负 `[1, 16, 64, 64]`銆?
5. runtime probe 鐜蹇収宸茶褰?Colab L4銆丆UDA 12.8銆丳ython 3.12.13銆乼orch 2.11.0+cu128銆乨iffusers 0.38.0銆乼ransformers 5.12.1銆乤ccelerate 1.14.0 鍜?huggingface_hub 1.20.1銆?
6. `paper_workflow/minimal_latent_injection_run.ipynb` 鎻愪緵鏈€灏?latent injection 鐨?Colab 鍐峰惎鍔ㄥ叆鍙? 棣栦釜浠ｇ爜鍗曞厓鍏堟寕杞?Google Drive, 榛樿 `SLM_WM_MODEL_SELECTION=auto`, 褰撳墠浠?SD3.5 Medium 涓绘ā鍨嬩负鏈€灏忕湡瀹炴敞鍏ラ獙璇佸璞? 骞跺皢 zip 闀滃儚鍒?`SLM/minimal_diffusion_latent_injection/`銆?
7. `paper_workflow/colab_utils/minimal_latent_injection.py` 鎵胯浇 clean / watermarked paired image 鐢熸垚銆乴atent callback 娉ㄥ叆銆乴atent update records銆乸aired quality metrics銆乪nvironment report銆乵anifest銆亃ip 鎵撳寘鍜?Google Drive 闀滃儚閫昏緫銆?
8. 宸插璁?`outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip`; 璇ュ寘瀵瑰簲鎻愪氦 `b2be25c`, ZIP 瀹屾暣鎬ч€氳繃, SHA-256 涓?`bff5f14c7e57e669dc6e9e371bb999fa663581bf4033ba771ab6595ff5d0ec0c`銆?
9. minimal latent injection 缁撴灉涓? SD3.5 Medium 鐢熸垚 clean / watermarked paired images銆? 鏉?latent update records銆乸aired quality metrics銆乵anifest 鍜?environment report; 璐ㄩ噺鎸囨爣璁板綍鍖呮嫭 PSNR `37.86754851645436`銆丼SIM `0.9987065282916542`銆丮SE `0.00016339740250259638` 鍜?mean_abs_error `0.00732430862262845`銆?
10. `configs/colab_sd35_runtime_constraints.txt` 璁板綍鏈宸查獙璇佺殑 SD3.5 Medium Colab 渚濊禆缁勫悎, 浠呬綔涓鸿繙绋?Notebook 澶嶇幇鍙傝€? 涓嶅睘浜庢湰鍦伴粯璁ゅ畨瑁呬緷璧栥€?
11. `tests/constraints/test_notebook_entrypoint_contract.py` 楠岃瘉 Notebook 鏂囦欢鍚嶃€佹湭淇濆瓨鎵ц杈撳嚭銆丯otebook 璋冪敤 repository helper銆乸robe / injection 浜х墿鍙鎵撳寘鍜岄暅鍍? 浠ュ強 Colab 杩愯鐜绾︽潫璁板綍涓嶅己鍒跺畨瑁呭钩鍙版彁渚涚殑 torch銆?
12. `tests/functional/test_minimal_latent_injection_helpers.py` 楠岃瘉鏈€灏?injection 閰嶇疆銆佺ǔ瀹氭憳瑕併€佽交閲忚川閲忔寚鏍囥€侀粯璁ゆā鍨嬮€夋嫨銆佽繍琛岀幆澧冪増鏈揩鐓у拰 environment report 鍐欏嚭銆?
13. `docs/field_registry.md` 宸茬櫥璁扮湡瀹?runtime probe銆乤rchive 鍜屾渶灏?latent injection 鏂板瀛楁銆?

### stage04 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄湡瀹?SD3.5 / SD3 鎺ㄧ悊閾捐矾銆佺湡瀹?latent trajectory 鎹曡幏鍜?SD3.5 Medium 鏈€灏?latent injection 宸ョ▼楠岃瘉銆?
2. 褰撳墠 `supports_paper_claim=false` 鐨勮竟鐣屼繚鎸佷笉鍙? 杩欎簺缁撴灉涓嶅緱鐩存帴浣滀负璁烘枃涓殑 watermark detection銆乺obustness 鎴?fixed-FPR 缁撹銆?
3. 褰撳墠闃舵涓嶈姹傜湡瀹?attention capture; Q/K attention 鎴栧彲瀹¤ attention map 搴斿湪鍚庣画 attention capture 涓撻棬鏋勫缓鍗曞厓涓帴鍏ャ€?
4. SD3.5 Medium 鏄悗缁富绾挎ā鍨? SD3 Medium 浠呬繚鐣欎负鍏煎鎬?fallback 涓庡鐓ц瘉鎹€?

### stage04 褰撳墠楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 7 passed |
| `pytest tests/functional/test_minimal_latent_injection_helpers.py -q` | pass, 7 passed |
| `pytest -q` | pass, 38 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_05_colab_drive_workflow

| item | value |
| --- | --- |
| construction_unit_name | `stage_05_colab_drive_workflow` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/core_package_boundary_freeze/manifest.local.json`; `outputs/algorithm_primitives/manifest.local.json`; `outputs/core_method_synthetic_smoke/manifest.local.json`; `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/colab_drive_workflow/manifest.local.json`; `GoogleDrive/SLM/colab_drive_workflow/manifest.json`; `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip` |
| expected_outputs | `paper_workflow/colab_drive_cold_start_smoke.ipynb`; `paper_workflow/drive_manifest_reload_smoke.ipynb`; `paper_workflow/colab_utils/drive_paths.py`; `paper_workflow/colab_utils/dependency_check.py`; `paper_workflow/colab_utils/mount_drive.py`; `paper_workflow/colab_utils/runtime_setup.py`; `paper_workflow/colab_utils/manifest_io.py`; `paper_workflow/colab_utils/drive_workflow.py`; `scripts/colab_drive_entry.py`; `scripts/sync_local_outputs_to_drive.py`; `scripts/write_workflow_manifest.py`; `scripts/verify_drive_artifacts.py`; `outputs/colab_drive_workflow/colab_env_report.json`; `outputs/colab_drive_workflow/drive_mount_report.json`; `outputs/colab_drive_workflow/cold_start_smoke_record.jsonl`; `outputs/colab_drive_workflow/reload_smoke_record.jsonl`; `outputs/colab_drive_workflow/local_output_sync_report.json`; `outputs/colab_drive_workflow/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻?Colab 鏃犳硶鎸傝浇 Drive, 鏈湴鍛戒护浠呭啓鍏?`outputs/colab_drive_workflow/drive_mirror/` 闀滃儚鐩綍骞惰褰?`unsupported_reason`; 璇ラ暅鍍忎笉寰楁敮鎸佹寮忚鏂?claim銆?|
| invariants | Notebook 鍙綔涓哄叆鍙? Drive manifest銆侀暅鍍忎笌閲嶈浇鏍￠獙閫昏緫浣嶄簬 repository helper 鍜?scripts; `main/` 涓嶄緷璧?Colab銆丏rive 鎴?Notebook銆?|
| next_stage_entry | Drive manifest 鍦?Colab 涓敓鎴? 涓旈潪绌鸿緭鍏ョ櫥璁般€侀暅鍍忓拰 reload 鏍￠獙鍧囬€氳繃; 鍙繘鍏?`stage_06_prompt_split_records_protocol`銆?|

### stage05 宸插畬鎴愬唴瀹?

1. 鏂板 Colab Drive workflow helper, 灏嗚矾寰勮В鏋愩€佷緷璧栧揩鐓с€丏rive 鎸傝浇鎶ュ憡銆乵anifest 璇诲啓銆佹湰鍦?outputs 闀滃儚鍜?reload 鏍￠獙鍒嗙鍒?`paper_workflow/colab_utils/` 涓嬬殑璇箟鍖栨ā鍧椼€?
2. 鏂板 `scripts/colab_drive_entry.py`銆乣scripts/sync_local_outputs_to_drive.py`銆乣scripts/write_workflow_manifest.py` 鍜?`scripts/verify_drive_artifacts.py`, 浣滀负 Notebook 鍙皟鐢ㄧ殑浠撳簱鍏ュ彛銆?
3. 鏂板 `paper_workflow/colab_drive_cold_start_smoke.ipynb` 涓?`paper_workflow/drive_manifest_reload_smoke.ipynb`, 涓や釜 Notebook 鍧囦笉淇濆瓨鎵ц杈撳嚭, 涓斿彧璋冪敤 repository helper銆?
4. 鏂板杞婚噺娴嬭瘯瑕嗙洊鏈湴 outputs 闀滃儚銆乵anifest 鍐欏叆銆乺eload 鏍￠獙銆佹湰鍦拌緭鍑虹洰褰曠害鏉熴€丏rive 鎸傝浇璺宠繃鎶ュ憡鍜屼緷璧栧揩鐓ч潪 claim 杈圭晫銆?
5. 鏈湴鎵ц `python scripts/colab_drive_entry.py` 宸插湪 `outputs/colab_drive_workflow/` 鐢熸垚鍙璁?smoke 浜х墿, 骞堕獙璇佹湰鍦伴暅鍍?reload 閫氳繃銆?
6. 宸蹭慨姝?Colab 鍐峰惎鍔ㄨ緭鍏ヨ竟鐣? 鑻?clone 鍚庢湰鍦?`outputs/` 涓虹┖, workflow 浼氱櫥璁?Google Drive 涓凡鏈夌殑 `SLM/real_sd_runtime_probe/` 涓?`SLM/minimal_diffusion_latent_injection/` 鐪熷疄杩愯浜х墿, 鑰屼笉鏄妸绌?manifest 璇垽涓烘湁鏁堣瘉鎹€?
7. 宸插璁?`outputs/colab_drive_workflow-20260620T114217Z-3-001.zip`; ZIP 瀹屾暣鎬ч€氳繃, SHA-256 涓?`427f01ed221c26cc1ee319c6a45ffdd9ab35caccf96541b741af872dab0fcb98`銆?
8. 璇ョ粨鏋滃寘涓?`metadata.workflow_decision=pass`, `reload_decision=pass`, `verified_file_count=2`, `missing_input_count=0`, `digest_mismatch_count=0`銆?
9. 璇ョ粨鏋滃寘鐧昏浜?Google Drive 涓凡鏈夌殑鍓嶅簭鐪熷疄浜х墿: `SLM/minimal_diffusion_latent_injection/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` 鍜?`SLM/real_sd_runtime_probe/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`銆?
10. `docs/field_registry.md` 宸茬櫥璁?Colab Drive workflow 鏂板瀛楁銆?

### stage05 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄?Colab 涓?Google Drive 涔嬮棿鐨勯潪绌哄墠搴忎骇鐗╃櫥璁般€侀暅鍍忓拰閲嶈浇鏍￠獙銆?
2. 褰撳墠 `supports_paper_claim=false` 鐨勮竟鐣屼繚鎸佷笉鍙? 杩欎簺缁撴灉鍙綔涓?workflow provenance, 涓嶇洿鎺ヤ綔涓鸿鏂囦腑鐨?detection銆乺obustness 鎴?fixed-FPR 缁撹銆?
3. `drive_mount_report.json` 涓?`mount_decision=skipped`銆乣mounted=true`銆乣unsupported_reason=mount_not_requested` 琛ㄧず Notebook 宸查鍏堟寕杞?Drive, helper 鏈噸澶嶆墽琛屾寕杞藉姩浣? 涓嶆瀯鎴愰樆鏂」銆?

### stage05 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/colab_drive_entry.py` | pass, local_manifest_count=7, mirrored_file_count=18, reload_decision=pass |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 8 passed |
| `pytest tests/functional/test_colab_drive_workflow_helpers.py -q` | pass, 6 passed |
| `pytest -q` | pass, 43 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |
| `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip` | pass, SHA-256 `427f01ed221c26cc1ee319c6a45ffdd9ab35caccf96541b741af872dab0fcb98` |

## stage_06_prompt_split_records_protocol

| item | value |
| --- | --- |
| construction_unit_name | `stage_06_prompt_split_records_protocol` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/colab_drive_workflow-20260620T114217Z-3-001.zip`; `outputs/prompts.zip`; `configs/paper_main_probe_prompts.txt`; `configs/paper_main_pilot_prompts.txt`; `configs/paper_main_full_prompts.txt` |
| expected_output_manifest | `outputs/prompt_event_protocol/manifest.local.json` |
| expected_outputs | `configs/paper_main_probe_prompts.txt`; `configs/paper_main_pilot_prompts.txt`; `configs/paper_main_full_prompts.txt`; `outputs/prompt_event_protocol/prompt_records.jsonl`; `outputs/prompt_event_protocol/event_records.jsonl`; `outputs/prompt_event_protocol/prompt_manifest.json`; `outputs/prompt_event_protocol/split_manifest.json`; `outputs/prompt_event_protocol/event_protocol_manifest.json`; `outputs/prompt_event_protocol/prompt_statistics.json`; `outputs/prompt_event_protocol/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻?prompt bank銆乸rompt 閰嶇疆銆佸墠搴?Drive workflow 璇佹嵁鎴栧瓧娈电櫥璁扮己澶? 鍋滄鎺ㄨ繘骞朵慨澶嶅崗璁緭鍏? 涓嶅緱鎵嬪伐鏀瑰啓 prompt_id 鎴?event_id銆?|
| invariants | records 鍙兘鐢?`experiments/` 鎴?`scripts/` 鍐欏嚭; `main/` 涓嶅啓 records; calibration 涓?test 涓嶅叡浜?prompt_id; 褰撳墠鍗忚浜х墿涓嶆敮鎸佹寮忚鏂?claim銆?|
| next_stage_entry | prompt銆乻plit銆乻ample role 鍜?event manifest 鍧囧彲澶嶇幇, 鍙繘鍏?`stage_07_semantic_mask_risk_field_safe_subspace`銆?|

### stage06 宸插畬鎴愬唴瀹?

1. 浣跨敤 `outputs/prompts.zip` 閲嶆柊鐢熸垚椤圭洰 prompt 閰嶇疆; 杈撳叆 zip 鐨?SHA-256 涓?`197cb1c40d2ff131e761c70b56f41164c4e7ad168f35a63cb1c2bbe5c46e1eee`銆?
2. 鏂板 `scripts/import_prompt_bank.py`, 浠庡閮?prompt bank 璇诲彇 probe銆乸ilot 鍜?full 涓夌粍 prompt, 缁熶竴瑙勮寖鍖栫┖鐧? 骞舵浛鎹粨搴撴不鐞嗕笉鍏佽鍐欏叆閰嶇疆姝ｆ枃鐨勮繃绋嬫爣璁拌瘝銆?
3. `configs/paper_main_probe_prompts.txt`銆乣configs/paper_main_pilot_prompts.txt` 鍜?`configs/paper_main_full_prompts.txt` 褰撳墠鍒嗗埆鍖呭惈 10銆?00 鍜?6000 鏉?prompt銆?
4. prompt bank 瀵煎叆杩囩▼涓? pilot 涓?full 鍚勬湁 1 鏉?prompt 鍥犲懡鍚嶆不鐞嗙害鏉熻璇箟绛変环鏇挎崲涓?`concert platform` 琛ㄨ揪, probe 鏃犻渶鏇挎崲銆?
5. `experiments/protocol/prompts.py` 璐熻矗 prompt 鏂囨湰瑙勮寖鍖栥€佽涔夋爣绛炬淳鐢熴€侀闄╅厤缃淳鐢熴€佺ǔ瀹?`prompt_id` 鐢熸垚, 骞跺湪 prompt record 涓繚鐣?split 瀛楁銆?
6. `experiments/protocol/splits.py` 鍥哄畾 `dev`銆乣calibration`銆乣test` 涓変釜 split, 骞舵寜 prompt set 涓?risk profile 鍒嗗眰鍚庤繘琛岀ǔ瀹氬垝鍒? 閬垮厤 calibration/test 鍦?prompt_id 灞傞潰浜ゅ弶銆?
7. `experiments/protocol/events.py` 鐢?prompt 涓?sample role 鏋勯€?`positive_source`銆乣clean_negative` 鍜?`attacked_negative` 涓夌被浜嬩欢, 骞剁敓鎴愮ǔ瀹?`event_id`銆?
8. `experiments/protocol/records.py` 涓?`experiments/protocol/calibration.py` 璐熻矗 JSONL 鍐欏嚭銆佽交閲忓敮涓€鎬ф牎楠屽拰鍗忚缁熻鎽樿銆?
9. `scripts/write_prompt_event_protocol.py` 灏?prompt records銆乪vent records銆乸rompt manifest銆乻plit manifest銆乪vent protocol manifest銆乸rompt statistics 鍜屾湰鍦?manifest 鍐欏叆 `outputs/prompt_event_protocol/`, 骞跺湪 manifest 杈撳叆涓櫥璁?`outputs/prompts.zip`銆?
10. 褰撳墠鍗忚杈撳嚭 `prompt_count=6610`, `event_count=19830`, `split_counts` 涓?`dev=659`銆乣calibration=2970`銆乣test=2981`, 涓変釜 sample role 鍚?6610 鏉′簨浠躲€?
11. 褰撳墠鍗忚杈撳嚭 `calibration_test_disjoint=true`, `protocol_decision=pass`, `supports_paper_claim=false`銆?
12. `docs/field_registry.md` 宸茬櫥璁?prompt銆乻plit銆乪vent銆乻ample role銆乸rotocol manifest銆乸rompt bank 瀵煎叆鎽樿鍜岀粺璁℃憳瑕佺浉鍏冲瓧娈点€?
13. 鏂板 `tests/functional/test_prompt_bank_import.py` 涓?`tests/functional/test_prompt_event_protocol.py`, 瑕嗙洊 prompt bank 瀵煎叆銆佺ǔ瀹?ID銆乻plit 鏃犱氦鍙夈€佸彈娌荤悊杈撳嚭鐩綍鍜?manifest 鍐欏嚭杈圭晫銆?

### stage06 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄鏂囧疄楠屽崗璁储寮? 涓嶆槸姝ｅ紡妫€娴嬫寚鏍囥€侀瞾妫掓€ф寚鏍囨垨 fixed-FPR 缁撹銆?
2. `prompt_records.jsonl` 鍜?`event_records.jsonl` 鍙互浣滀负鍚庣画瀹為獙 runner 鐨勮緭鍏ョ储寮? 浣嗕笉寰楃洿鎺ヤ綔涓鸿鏂?claim 鏀拺璇佹嵁銆?
3. `calibration` 涓?`test` 鐨勭粺璁¤竟鐣屽湪 prompt_id 灞傞潰淇濇寔鏃犱氦鍙? 鍚庣画闃堝€兼牎鍑嗗繀椤荤户缁部鐢ㄨ繖涓€杈圭晫銆?
4. `dev` split 浠呯敤浜庡紑鍙戝拰閾捐矾妫€鏌? 涓嶅緱鐢ㄤ簬鍐荤粨 fixed-FPR 闃堝€兼垨 rescue gate銆?
5. `outputs/prompts.zip` 鏄湰娆?prompt bank 瀵煎叆鏉ユ簮, 涓嶅睘浜庡簲鎻愪氦鍒?Git 鐨勪粨搴撳唴瀹广€?

### stage06 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python scripts/import_prompt_bank.py` | pass, probe=10, pilot=600, full=6000, sanitized counts probe=0銆乸ilot=1銆乫ull=1 |
| `python scripts/write_prompt_event_protocol.py` | pass, prompt_count=6610, event_count=19830, calibration_test_disjoint=true |
| `pytest tests/functional/test_prompt_bank_import.py tests/functional/test_prompt_event_protocol.py -q` | pass, 5 passed |
| `pytest -q` | pass, 50 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_07_semantic_mask_risk_field_safe_subspace

| item | value |
| --- | --- |
| construction_unit_name | `stage_07_semantic_mask_risk_field_safe_subspace` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/prompt_event_protocol/manifest.local.json`; `outputs/prompt_event_protocol/prompt_records.jsonl`; `outputs/real_sd_runtime_probe_package_20260620t10451781952321z_b2be25c.zip`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/semantic_subspace/manifest.local.json` |
| expected_outputs | `outputs/semantic_subspace/semantic_route_records.jsonl`; `outputs/semantic_subspace/subspace_plan_records.jsonl`; `outputs/semantic_subspace/mask_projection_reports/mask_projection_reports.jsonl`; `outputs/semantic_subspace/basis_digests.json`; `outputs/semantic_subspace/semantic_subspace_summary.json`; `outputs/semantic_subspace/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻ョ湡瀹?latent trace 鎽樿鍖呬笉鍙敤, 浣跨敤纭畾鎬?lightweight latent reference 缁х画楠岃瘉璇箟鎺╃爜褰卞搷 feature operator 涓?basis, 涓斾繚鎸?`supports_paper_claim=false`銆?|
| invariants | saliency銆乻egmentation 鍜?SD attention capture 涓嶈繘鍏?`main/`; `main/` 鍙帴鏀舵爣鍑嗗寲 mask銆乴atent mask 鍜?feature tensor; 鏃犺涔夋帺鐮佽矾寰勫彧浣滀负娑堣瀺鎴栬瘖鏂矾寰勩€?|
| next_stage_entry | semantic route銆乵ask projection銆乤pproximate JVP 鍜?safe basis 鍧囨湁 digest, 涓旇涔夋帺鐮佷細鏀瑰彉 basis; 鍙繘鍏?`stage_08_lf_hf_content_carriers`銆?|

### stage07 宸插畬鎴愬唴瀹?

1. 鏂板 `main/methods/semantic/risk_field.py`, 瀹炵幇鏍囧噯鍖栬涔夈€佺汗鐞嗐€佺ǔ瀹氭€у拰鏄捐憲鎬у悜閲忓埌椋庨櫓鍦轰笌鎵胯浇棰勭畻鐨勬槧灏勩€?
2. 鏂板 `main/methods/semantic/latent_mask.py`, 瀹炵幇 `M_z = Pi_{x->z}(M_x)` 鐨勮交閲忔姇褰卞拰 `M_z * z_t` 鎺╃爜浣滅敤銆?
3. 鏂板 `main/methods/semantic/routing.py`, 鏍规嵁椋庨櫓鍦轰笌 latent mask 鐢熸垚 LF銆丠F 鍜?attention 鍊欓€夎酱璺敱銆?
4. 鏂板 `main/methods/subspace/trajectory_features.py`, 瀹炵幇 `P^T vec(Norm(M_z * z_t))` 鐨勮交閲?feature operator銆?
5. 鏂板 `main/methods/subspace/jvp_estimator.py`, 鐢ㄧ浉閭诲樊鍒嗗疄鐜板彲瀹¤ approximate JVP 鎽樿銆?
6. 鏂板 `main/methods/subspace/safe_basis.py` 鍜?`main/methods/subspace/route_projection.py`, 瀹炵幇 semantic safe basis銆乶o semantic mask銆乬lobal nullspace 鍜?diagnostic basis 鍥涚鍙繍琛屽熀搴曠瓥鐣? 骞剁敓鎴?route projection digest銆?
7. 鏂板 `scripts/write_semantic_subspace_outputs.py`, 浠?prompt protocol records 涓庣湡瀹?SD3.5 latent trace 鎽樿鍖呬腑鏋勯€?semantic route records銆乻ubspace plan records銆乵ask projection reports銆乥asis digests銆乻ummary 鍜?manifest銆?
8. 褰撳墠 `outputs/semantic_subspace/semantic_subspace_summary.json` 鏄剧ず `semantic_route_record_count=6610`, `subspace_plan_record_count=6610`, `mask_projection_report_count=6610`, `unique_route_digest_count=6610`, `semantic_mask_changed_basis_count=6610`, `protocol_decision=pass`銆?
9. 褰撳墠 `supports_paper_claim=false` 杈圭晫淇濇寔涓嶅彉; 鏈樁娈典骇鐗╄瘉鏄庢満鍒堕摼璺彲瀹¤, 涓嶇洿鎺ヤ綔涓?detection 鎴?fixed-FPR 璁烘枃缁撹銆?
10. 鏂板 `tests/functional/test_semantic_subspace.py`, 瑕嗙洊涓嶅悓璇箟鎺╃爜浜х敓涓嶅悓 route銆佸叧闂涔夋帺鐮佹敼鍙?basis銆佹秷铻嶅熀搴曞彲杩愯銆佽剼鏈緭鍑?manifest 鍜岃緭鍑虹洰褰曠害鏉熴€?
11. `docs/field_registry.md` 宸茬櫥璁版湰闃舵鏂板 route銆乵ask銆乫eature operator銆乤pproximate JVP銆乥asis strategy銆乥asis digest 鍜?summary 瀛楁銆?

### stage07 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄牳蹇冩柟娉曞眰鐨勬爣鍑嗗寲 semantic mask銆乺isk field銆乫eature operator銆乤pproximate JVP 鍜?semantic safe basis, 涓嶆槸姝ｅ紡 SD attention capture 鎴栬鏂囦富瀹為獙缁熻銆?
2. runtime 灞備粛璐熻矗 saliency銆乻egmentation銆乸redicted x0 涓?attention capture; core 鏂规硶灞備笉鍔犺浇妯″瀷鏉冮噸銆?
3. `no_semantic_mask`銆乣global_nullspace` 鍜?`diagnostic_basis` 浠呬綔涓烘秷铻嶆垨璇婃柇璺緞, 涓嶅緱浼鎴?SLM-WM 涓绘柟娉曘€?
4. 鍚庣画 LF/HF carrier 鏋勫缓搴旇鍙?`subspace_plan_records.jsonl` 涓?`basis_digests.json`, 骞朵繚鐣?calibration/test split 杈圭晫銆?

### stage07 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python scripts/write_semantic_subspace_outputs.py` | pass, semantic_route_record_count=6610, subspace_plan_record_count=6610, semantic_mask_changed_basis_count=6610 |
| `pytest tests/functional/test_semantic_subspace.py -q` | pass, 4 passed |
| `pytest -q` | pass, 54 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_08_lf_hf_content_carriers

| item | value |
| --- | --- |
| construction_unit_name | `stage_08_lf_hf_content_carriers` |
| phase_status | `completed` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/semantic_subspace/manifest.local.json`; `outputs/semantic_subspace/subspace_plan_records.jsonl`; `outputs/semantic_subspace/semantic_route_records.jsonl`; `outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip` |
| expected_output_manifest | `outputs/content_carriers/manifest.local.json` |
| expected_outputs | `outputs/content_carriers/content_detection_records.jsonl`; `outputs/content_carriers/lf_hf_score_table.csv`; `outputs/content_carriers/paired_quality_metrics.csv`; `outputs/content_carriers/content_score_distribution.csv`; `outputs/content_carriers/content_carrier_summary.json`; `outputs/content_carriers/manifest.local.json` |
| blocking_items | 鏃犮€?|
| fallback_path | 鑻ヨ涔夊瓙绌洪棿 records 鎴栫湡瀹炴渶灏忔敞鍏ヨ川閲忓寘涓嶅彲鐢? 鍋滄鎺ㄨ繘骞朵慨澶嶅墠搴忚緭鍏? 涓嶅厑璁哥敤鎵嬪伐闃堝€兼姇绁ㄦ垨鏈櫥璁版枃浠舵浛浠ｅ唴瀹瑰垎鏁伴摼璺€?|
| invariants | LF 涓哄唴瀹逛富璇佹嵁, HF 浠呬负琛ュ厖璇佹嵁; 涓嶄负 LF/HF 鍒嗗埆璁剧疆鐙珛姝ｅ垽闃堝€煎悗鎶曠エ; 褰撳墠浜х墿淇濇寔 `supports_paper_claim=false`, 涓嶈兘鐩存帴浣滀负璁烘枃 fixed-FPR 鎴?robustness 缁撹銆?|
| next_stage_entry | 鍐呭杞戒綋 records銆佺粺涓€鍐呭鍒嗘暟銆佹満鍒跺紑鍏虫憳瑕佷笌 manifest 鍧囧彲閲嶅缓, 鍙繘鍏?`stage_09_self_attention_graph_geometry`銆?|

### stage08 宸插畬鎴愬唴瀹?

1. 鏂板 `main/methods/carrier/lf.py`, 瀹炵幇绋冲畾 LF 鍐呭妯℃澘銆佷綆棰戝钩婊戝拰 latent update 娲剧敓銆?
2. 鏂板 `main/methods/carrier/hf.py`, 瀹炵幇绋冲畾 HF 鍐呭妯℃澘銆乼ail truncation 鍜屽叧闂?tail truncation 鐨勬満鍒惰矾寰勩€?
3. 鏂板 `main/methods/carrier/compose.py`, 缁熶竴缁勫悎 `full_content_chain`銆乣lf_only`銆乣hf_only`銆乣no_hf`銆乣no_tail_truncation` 鍜?`no_lf` 鍏被鍐呭鏈哄埗寮€鍏炽€?
4. 鏂板 `main/methods/detection/scores.py` 鍜?`main/methods/detection/fusion.py`, 瀹炵幇 `s_c = lambda_LF s_LF + lambda_HF s_HF`, 涓?`lambda_LF > lambda_HF`, `used_independent_branch_vote=false`銆?
5. 鏂板 `scripts/write_content_carrier_outputs.py`, 浠庤涔夊瓙绌洪棿 records 涓庢渶灏?latent injection 璐ㄩ噺鍖呴噸寤哄唴瀹规娴?records銆丩F/HF score table銆乸aired quality metrics銆乻core distribution銆乻ummary 鍜?manifest銆?
6. 褰撳墠 `outputs/content_carriers/content_carrier_summary.json` 鏄剧ず `content_detection_record_count=19830`, `score_count=19830`, `fixed_fpr_ready=true`, `used_independent_branch_vote=false`, `protocol_decision=pass`, `supports_paper_claim=false`銆?
7. 鏂板 `tests/functional/test_content_carriers.py`, 瑕嗙洊 LF/HF 杞戒綋鎽樿绋冲畾鎬с€佹満鍒跺紑鍏崇湡瀹炴敼鍙?update銆佺粺涓€鍐呭鍒嗘暟 fixed-FPR 杈圭晫銆佸啓鍑鸿剼鏈?manifest 鍜?outputs 鐩綍绾︽潫銆?
8. `docs/field_registry.md` 宸茬櫥璁板唴瀹硅浇浣撱€佸唴瀹瑰垎鏁般€佹満鍒跺紑鍏炽€乻core distribution 鍜?summary 鐩稿叧瀛楁銆?

### stage08 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄牳蹇冩柟娉曞眰 LF/HF 鍐呭杞戒綋鍜岀粺涓€鍐呭鍒嗘暟鏈哄埗, 涓嶆槸鏈€缁堣鏂囬槇鍊兼牎鍑嗐€乤ttack matrix 鎴栨寮忓浐瀹?FPR 瀹為獙缁撹銆?
2. `fixed_fpr_ready=true` 浠呰〃绀哄唴瀹瑰垎鏁拌褰曚繚鐣欎簡鍙繘鍏ュ悗缁?fixed-FPR calibration 鐨勭粺璁″舰鎬? 鐪熷疄闃堝€煎喕缁撳繀椤荤户缁娇鐢?calibration split, 骞朵笖涓嶈兘涓?test split 娣风敤銆?
3. `rescue` 涓嶅湪鏈樁娈佃Е鍙戞鍒? 鍚庣画鍑犱綍 rescue 蹇呴』鍦ㄥ悓涓€ fixed-FPR 缁熻杈圭晫鍐呭璁? 涓嶈兘鏂板鐙珛闃虫€ч€氶亾銆?
4. LF-only銆丠F-only銆丯o-HF銆丯o-tail-truncation 鍜?No-LF 鍧囦綔涓烘満鍒惰瘖鏂垨娑堣瀺璺緞, 涓嶅緱浼涓?SLM-WM 涓绘柟娉曘€?

### stage08 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_content_carrier_outputs.py` | pass, content_detection_record_count=19830, score_count=19830, protocol_decision=pass |
| `pytest tests/functional/test_content_carriers.py -q` | pass, 5 passed |
| `pytest -q` | pass, 59 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_09_self_attention_graph_geometry

| item | value |
| --- | --- |
| construction_unit_name | `stage_09_self_attention_graph_geometry` |
| phase_status | `real_capture_workflow_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/content_carriers/manifest.local.json`; `outputs/content_carriers/content_carrier_summary.json`; `outputs/sd_runtime_adapter/manifest.local.json`; `outputs/sd_runtime_adapter/attention_capture_records.jsonl`; Colab 杩愯鍚庡彲鏇挎崲涓?`outputs/real_attention_geometry/real_attention_capture_records.jsonl` |
| expected_output_manifest | `outputs/attention_geometry/manifest.local.json`; `outputs/real_attention_geometry/real_attention_manifest.local.json`; `outputs/real_attention_geometry/attention_geometry_archive_manifest.local.json` |
| expected_outputs | `paper_workflow/attention_geometry_capture_run.ipynb`; `paper_workflow/colab_utils/attention_geometry_capture.py`; `outputs/real_attention_geometry/real_attention_capture_records.jsonl`; `outputs/real_attention_geometry/real_attention_capture_summary.json`; `outputs/real_attention_geometry/real_attention_environment_report.json`; `outputs/attention_geometry/attention_graph_records.jsonl`; `outputs/attention_geometry/geometry_evidence_records.jsonl`; `outputs/attention_geometry/attention_relation_consistency.csv`; `outputs/attention_geometry/geometry_evidence_summary.json`; `outputs/attention_geometry/manifest.local.json`; `outputs/real_attention_geometry/attention_geometry_package_<utc>_<short_commit>.zip`; `GoogleDrive/SLM/attention_geometry/attention_geometry_package_<utc>_<short_commit>.zip` |
| blocking_items | 鏈湴鐜鏃?GPU 鍜岀湡瀹?SD3.5 Medium 鏉冮噸, 鍥犳鏈湴榛樿浜х墿浠嶆潵鑷墠搴?synthetic attention capture; 鐪熷疄 `attention_geometry_ready=true` 闇€瑕佽繍琛?Colab Notebook 骞跺洖浼犵粨鏋滃寘瀹¤銆?|
| fallback_path | 鑻ョ湡瀹?attention hook 涓嶅彲鐢? Notebook 浼氳 `attention_geometry_ready` 鏂█澶辫触, 骞朵繚鐣欏け璐?summary; 涓嶅厑璁告妸 synthetic attention capture 鏀瑰啓涓虹湡瀹?capture銆?|
| invariants | 鍑犱綍璇佹嵁鍙褰曞彲闈犳€х粺璁? `direct_positive_decision=false`; 鍙湁鎵€鏈?capture records 鍧囦负鐪熷疄鍙璁¤褰曘€乣real_attention_capture_count>0` 涓?`unsupported_capture_count=0` 鏃? `attention_geometry_ready` 鎵嶈兘涓?true銆?|
| next_stage_entry | 杩愯骞跺璁?`attention_geometry_package_<utc>_<short_commit>.zip` 鍚? 鑻?summary 鏄剧ず `attention_geometry_ready=true`, 鎵嶅厑璁告妸鐪熷疄 attention-relative latent update 浣滀负鍚庣画鏂规硶瀹炵幇杈撳叆銆?|

### stage09 宸插畬鎴愬唴瀹?

1. 鏂板 `main/methods/geometry/attention_graph_types.py`, 瀹氫箟 attention graph record 涓?geometry evidence record 鐨?typed object銆?
2. 鏂板 `main/methods/geometry/recovery.py`, 瀹炵幇 `softmax(QK^T / sqrt(d))`銆佺ǔ瀹?token 闆嗛€夋嫨銆佺浉瀵瑰叧绯绘娊鍙栥€乤nchor graph digest 鍜屽嚑浣曟仮澶嶇粺璁°€?
3. 鏇存柊 `experiments/runtime/diffusion/attention_capture.py`, 澧炲姞浠?Q/K 鍚戦噺鏋勯€犲彲瀹¤ attention capture record 鐨勭函鍑芥暟鍏ュ彛, 淇濇寔鐪熷疄 runtime hook 涓庢牳蹇冩柟娉曞眰瑙ｈ€︺€?
4. 鏇存柊 `scripts/write_attention_geometry_outputs.py`, 鏀寔閫氳繃 `--attention-records-path` 鎸囧悜鐪熷疄 Colab capture records; 鍙湁 records 鍏ㄩ儴鏃?`unsupported_reason`銆乣metadata.capture_is_synthetic=false`銆佸寘鍚湁鐣?`attention_matrix_preview`, 涓?`real_attention_capture_count>0`, summary 涓?`attention_geometry_ready` 鎵嶈兘涓?true銆?
5. 鏂板 `paper_workflow/colab_utils/attention_geometry_capture.py`, 鍦ㄧ湡瀹?SD3.5 Medium pipeline 鐨?transformer attention 妯″潡涓婃敞鍐?hook, 浠庣湡瀹?hidden states 鏋勯€犳湁鐣屽彲瀹¤ attention map, 鍐欏嚭鐪熷疄 capture records, 骞惰皟鐢ㄥ嚑浣曢噸寤鸿剼鏈埛鏂?`outputs/attention_geometry/`銆?
6. 鏂板 `paper_workflow/attention_geometry_capture_run.ipynb`, 鏀寔 Colab 鍐峰惎鍔? 鎸傝浇 Google Drive銆佹媺鍙栦唬鐮併€佸畨瑁呬緷璧栥€佽鍙?`HF_TOKEN`銆佸姞杞?SD3.5 Medium銆佹墽琛岀湡瀹?attention capture銆佹柇瑷€ `attention_geometry_ready=true`, 骞舵墦鍖呴暅鍍忓埌 `GoogleDrive/SLM/attention_geometry/`銆?
7. 鎵撳寘閫昏緫浼氭妸鐪熷疄 capture records銆佺湡瀹?capture summary銆佽繍琛岀幆澧冩姤鍛娿€乤ttention geometry records銆乻ummary銆乵anifest銆佽緭鍏ユ牳瀵?manifest 绛夋枃浠剁撼鍏?zip, 閬垮厤鍙笂浼犲崟涓€ summary銆?
8. 鏂板鍜屾洿鏂版祴璇曡鐩?Q/K 娉ㄦ剰鍔涘叕寮忋€佺湡瀹?preview 鐭╅樀 ready gate銆丯otebook 鍏ュ彛濂戠害銆佹墦鍖呴暅鍍忓绾﹀拰 outputs 鐩綍绾︽潫銆?
9. `docs/field_registry.md` 宸茬櫥璁扮湡瀹?attention map preview銆乤ttention records path銆佹崟鑾?tensor 褰㈢姸銆佸嚑浣?manifest / summary 璺緞鍜屽帇缂╁寘杈撳叆 manifest 鐩稿叧瀛楁銆?

### stage09 褰撳墠瀹屾垚杈圭晫

1. 鏈湴榛樿 `outputs/attention_geometry/geometry_evidence_summary.json` 浠嶄娇鐢ㄥ墠搴?synthetic attention capture, 鍥犳 `real_attention_capture_count=0`, `unsupported_capture_count=4`, `attention_geometry_ready=false`銆?
2. 鏂?Notebook 鐨勫畬鎴愬垽瀹氭槸寮烘柇瑷€: 鑻ョ湡瀹?SD3.5 Medium 鎺ㄧ悊娌℃湁鐢熸垚鏃?unsupported reason 鐨?capture records, Notebook 浼氬け璐? 涓嶄細浼€?ready 鐘舵€併€?
3. `attention_geometry_ready=true` 鐨勫敮涓€鏈夋晥璺緞鏄? Colab GPU 杩愯鐪熷疄 SD3.5 Medium -> 鍐欏嚭 `outputs/real_attention_geometry/real_attention_capture_records.jsonl` -> 鐢ㄨ records 閲嶅缓 `outputs/attention_geometry/` -> summary 婊¤冻鎵€鏈?records 鍧囦负鐪熷疄鍙璁¤褰曘€乣real_attention_capture_count>0` 涓?`unsupported_capture_count=0`銆?
4. 鍑犱綍璇佹嵁浠嶄笉寰楃洿鎺ョ粰鍑?positive 鍒ゅ畾; 鍚庣画鐪熷疄 attention-relative latent update 蹇呴』璇诲彇宸茬粡 ready 鐨?attention geometry 浜х墿銆?

### stage09 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attention_geometry_outputs.py` | pass, 榛樿 synthetic 杈撳叆涓?`attention_geometry_ready=false` |
| `pytest tests/functional/test_attention_geometry.py -q` | pass, 5 passed |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 10 passed |
| `pytest -q` | pass, 66 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_10_attention_relative_latent_update

| item | value |
| --- | --- |
| construction_unit_name | `stage_10_attention_relative_latent_update` |
| phase_status | `real_injection_audited` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip`; `outputs/semantic_subspace/manifest.local.json`; `outputs/content_carriers/manifest.local.json` |
| expected_output_manifest | `outputs/attention_latent_update/manifest.local.json`; Colab 杩愯鍚庝负 `outputs/attention_latent_injection/attention_latent_injection_manifest.local.json` |
| expected_outputs | `outputs/attention_latent_update/attention_carrier_records.jsonl`; `outputs/attention_latent_update/attention_update_stability.csv`; `outputs/attention_latent_update/attention_update_quality_metrics.csv`; `outputs/attention_latent_update/attention_update_summary.json`; `outputs/attention_latent_update/manifest.local.json`; `paper_workflow/attention_latent_injection_run.ipynb`; `paper_workflow/colab_utils/attention_latent_injection.py`; Colab 杩愯鍚庝负 `outputs/attention_latent_injection/attention_latent_injection_result.json`; `outputs/attention_latent_injection/attention_latent_update_records.jsonl`; `outputs/attention_latent_injection/attention_paired_quality_metrics.csv`; `outputs/attention_latent_injection/attention_injection_environment_report.json`; `outputs/attention_latent_injection/attention_latent_injection_manifest.local.json`; `GoogleDrive/SLM/attention_latent_injection/attention_latent_injection_package_<utc>_<short_commit>.zip` |
| blocking_items | 鐪熷疄 Colab GPU 缁撴灉鍖呭凡鍥炰紶骞跺畬鎴愭湰鍦板璁? `image_quality_metrics_ready=true`; 浣?`full_method_claim_ready=false`, 鍥犱负 fixed-FPR 涓?rescue 缁熻杈圭晫灏氭湭鍐荤粨銆?|
| fallback_path | 鑻ュ嚑浣曡瘉鎹笉鍙潬鎴?update 绋冲畾鎬ц竟鐣屼笉婊¤冻, carrier 鑷姩闄嶇骇涓?`evidence_only`, 鍙繚鐣欏嚑浣曡瘉鎹? 涓嶅啓鍏?Full 鏂规硶涓诲紶銆?|
| invariants | 鍑犱綍閾句笉鐩存帴 positive; attention update 鍙湪 `attention_geometry_ready=true` 涓斿嚑浣曡瘉鎹彲闈犳椂 active; 鏈湴璐ㄩ噺浠呬负 proxy, 涓嶆浛浠ｇ湡瀹?paired image 璐ㄩ噺鎸囨爣銆?|
| next_stage_entry | 宸插厑璁告妸鐪熷疄 attention latent injection 鍖呬綔涓?same-threshold geometric rescue 鐨勮緭鍏? `full_method_claim_ready` 浠嶉渶鍚庣画 fixed-FPR 涓?rescue 閾捐矾鍏卞悓纭銆?|

### stage10 宸插畬鎴愬唴瀹?

1. 鏂板 `main/methods/carrier/attention.py`, 瀹氫箟 `AttentionRelativeCarrier`, 鍏崇郴鎹熷け銆佸叧绯绘搴︽姇褰便€乤ctive update 涓?`evidence_only` 闄嶇骇杈圭晫銆?
2. 鏇存柊 `main/methods/carrier/__init__.py`, 瀵煎嚭 attention-relative carrier 鏂规硶鍏ュ彛銆?
3. 鏂板 `scripts/write_attention_latent_update_outputs.py`, 鍙粠 ready attention geometry zip 鎴栨湰鍦?ready 鐩綍璇诲彇鍥句笌鍑犱綍璇佹嵁, 缁撳悎 semantic safe subspace records 鐢熸垚 attention carrier records銆佸己搴︾ǔ瀹氭€ц〃銆佽川閲忎唬鐞嗚〃銆乻ummary 鍜?manifest銆?
4. 鏂板 `tests/functional/test_attention_latent_update.py`, 瑕嗙洊鍙潬鍑犱綍璇佹嵁瑙﹀彂 active update銆佷笉鍙潬鍑犱綍璇佹嵁闄嶇骇涓?`evidence_only`, 浠ュ強鑴氭湰浠?ready geometry 鍖呴噸寤哄彈娌荤悊浜х墿銆?
5. `docs/field_registry.md` 宸茬櫥璁?attention-relative carrier銆佸叧绯绘崯澶便€佸己搴︾ǔ瀹氭€с€佽川閲忎唬鐞嗗拰 Full 鏂规硶 claim 杈圭晫鐩稿叧瀛楁銆?
6. 鏂板 `paper_workflow/colab_utils/attention_latent_injection.py`, 鏀寔浠?Google Drive 璇诲彇鏈€鏂?ready attention geometry 鍖? 閲嶅缓 prompt / semantic / content / attention update 杈撳叆閾? 閫夋嫨 active carrier, 骞跺湪鐪熷疄 SD3.5 latent callback 涓墽琛?attention-relative update銆?
7. 鏂板 `paper_workflow/attention_latent_injection_run.ipynb`, 鏀寔 Colab 鍐峰惎鍔ㄣ€佹寕杞?Drive銆佽鍙?`HF_TOKEN`銆佹鏌?GPU銆佹墽琛岀湡瀹?attention latent injection銆佸己鏂█鐪熷疄 latent update 涓庤川閲忔寚鏍囧瓨鍦? 骞舵墦鍖呴暅鍍忓埌 `GoogleDrive/SLM/attention_latent_injection/`銆?
8. 鏇存柊 `tests/constraints/test_notebook_entrypoint_contract.py`, 瑕嗙洊鏂?Notebook 鍏ュ彛濮旀墭銆佹棤鎵ц杈撳嚭鍜岀湡瀹?injection 浜х墿鎵撳寘闀滃儚銆?

### stage10 褰撳墠浜х墿鎽樿

1. 褰撳墠杈撳叆浣跨敤鐪熷疄 SD3.5 Medium attention geometry 鍖?`outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip`, 鍏朵腑 `attention_geometry_ready=true`銆?
2. `outputs/attention_latent_update/attention_update_summary.json` 鏄剧ず `attention_carrier_record_count=64`, `active_update_count=16`, `evidence_only_count=48`, `attention_update_stable_count=16`, `protocol_decision=pass`銆?
3. 宸插璁＄湡瀹炵粨鏋滃寘 `outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip`, SHA256 涓?`c34577f71e549b6cf0dda43ed3dc8a582a45073f36b269d40cf454d598402b48`銆?
4. 鐪熷疄缁撴灉鍖呮樉绀?`run_decision=pass`, `latent_update_count=3`, 娉ㄥ叆姝ヤ负 `6, 10, 14`, `image_quality_metrics_ready=true`, PSNR 涓?`35.18531747817406`, SSIM 涓?`0.9976578187804996`銆?
5. `full_method_claim_ready=false` 浠嶄繚鎸佷笉鍙? 琛ㄧず灏氫笉鑳藉０绉?fixed-FPR 瀹屾暣鏂规硶涓诲紶宸茬粡瀹屾垚銆?

### stage10 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attention_latent_update_outputs.py --attention-geometry-package-path outputs/attention_geometry_package_20260620t13511781963497z_b237bb3.zip` | pass, `active_update_count=16`, `evidence_only_count=48` |
| `pytest tests/functional/test_attention_latent_update.py -q` | pass, 3 passed |
| `pytest tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 11 passed |
| `pytest -q` | pass, 70 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_11_same_threshold_geometric_rescue

| item | value |
| --- | --- |
| construction_unit_name | `stage_11_same_threshold_geometric_rescue` |
| phase_status | `local_rescue_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/content_carriers/manifest.local.json`; `outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip` |
| expected_output_manifest | `outputs/geometric_rescue/manifest.local.json` |
| expected_outputs | `outputs/geometric_rescue/aligned_detection_records.jsonl`; `outputs/geometric_rescue/rescue_metrics_summary.csv`; `outputs/geometric_rescue/content_failed_subset_summary.csv`; `outputs/geometric_rescue/geometry_rescue_audit.json`; `outputs/geometric_rescue/manifest.local.json` |
| blocking_items | 褰撳墠涓烘湰鍦板彈娌荤悊鏈哄埗璁板綍, aligned content score 浠嶆槸鐢卞嚑浣曞彲闈犳€с€佽竟鐣岃窛绂诲拰鏍锋湰瑙掕壊娲剧敓鐨勮交閲忎唬鐞? 鍚庣画鑻ヨ褰㈡垚姝ｅ紡璁烘枃涓诲紶, 闇€瑕佸湪鐪熷疄 aligned latent 涓婇噸鏂拌繍琛屽唴瀹规娴? 骞跺湪 calibration split 涓喕缁撳畬鏁?evidence-level 鍗忚銆?|
| fallback_path | 鑻ュ嚑浣曚笉鍙潬銆佸唴瀹瑰垎鏁颁笉鍦ㄨ竟鐣屽け璐ョ獥鍙? 鎴?fail reason 涓嶅睘浜?`geometry_suspected` / `low_confidence`, 鍒欎笉瑙﹀彂 rescue; `geo_direct_positive_audit` 鍙綔涓哄弽渚嬪璁? 涓嶈繘鍏ユ寮忔柟娉曘€?|
| invariants | 鍑犱綍閾句笉寰楃洿鎺?positive; rescue 鍚庝粛澶嶇敤鍚屼竴涓?`content_threshold=0.75`; 褰撳墠 `supports_paper_claim=false` 涓?`full_method_claim_ready=false`銆?|
| next_stage_entry | 鍙互杩涘叆 fixed-FPR calibration 涓庢寚鏍囧喕缁撴瀯寤? 涓嬩竴姝ュ繀椤诲悓鏃跺璁?raw content clean FPR銆乺escue 鍚?clean negative FPR 鍜?rescue 鍚?attacked negative FPR銆?|

### stage11 宸插畬鎴愬唴瀹?

1. 鏇存柊 `main/methods/detection/fusion.py`, 鏂板 `SameThresholdRescueConfig`銆乣GeometricRescueDecisionRecord`銆乣decide_same_threshold_geometric_rescue` 涓庢秷铻嶆ā寮忎笅鐨勫嚑浣曞彲闈犳€ч€夋嫨閫昏緫銆?
2. 鏇存柊 `main/methods/geometry/recovery.py`, 鏂板 `estimate_aligned_content_score` 杞婚噺浠ｇ悊鍏ュ彛, 鐢ㄤ簬鏈湴鍙楁不鐞嗚褰? 鍚庣画鐪熷疄 aligned latent 鍐呭妫€娴嬪彲鏇挎崲璇ュ叆鍙ｃ€?
3. 鏂板 `scripts/write_geometric_rescue_outputs.py`, 浠庣湡瀹?attention latent injection 鍖呭拰鍐呭妫€娴?records 閲嶅缓 aligned detection records銆乺escue metrics銆佸唴瀹瑰け璐ュ瓙闆嗘憳瑕併€乬eometry rescue audit 鍜?manifest銆?
4. 鏂板 `tests/functional/test_geometric_rescue.py`, 瑕嗙洊鍚岄槇鍊?rescue銆乶o-rescue 闃绘柇銆乬eo-direct-positive 鍙嶄緥瀹¤浠ュ強鍙楁不鐞嗕骇鐗╅噸寤恒€?
5. `docs/field_registry.md` 宸茬櫥璁?aligned detection銆乺escue 娑堣瀺銆乺escue gain銆乧lean / attacked FPR 涓?geo-direct-positive audit 瀛楁銆?

### stage11 褰撳墠浜х墿鎽樿

1. `outputs/geometric_rescue/geometry_rescue_audit.json` 鏄剧ず `protocol_decision=pass`, `attention_geometry_ready=true`, `image_quality_metrics_ready=true`, `latent_update_count=3`銆?
2. 褰撳墠鏈湴閲囨牱 `max_content_records=96`, 鐢熸垚 `aligned_detection_record_count=576`, 鍏朵腑 full-rescue 妯″紡璁板綍鏁颁负 `96`, `full_rescue_applied_count=1`銆?
3. full-rescue 妯″紡涓?`raw_content_clean_fpr=0.0`, `evidence_clean_fpr=0.0`, `evidence_attacked_fpr=0.03125`; 杩欎簺缁熻鍙綔涓哄悗缁?fixed-FPR 鏋勫缓杈撳叆, 涓嶈兘鏇夸唬姝ｅ紡 calibration銆?
4. `geo_direct_positive_audit_rate=0.5625` 鏄剧ず鍑犱綍鐩存帴鍒ゆ瀵?clean negative 鍏锋湁鏄庢樉 FPR 椋庨櫓, 鍥犳璇ュ垎鏀户缁繚鎸佷负鍙嶄緥瀹¤, 涓嶈繘鍏ユ寮忔柟娉曘€?
5. 鎵€鏈夋柊澧炰骇鐗╀繚鎸?`supports_paper_claim=false`, `full_method_claim_ready=false`銆?

### stage11 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_geometric_rescue_outputs.py --attention-injection-package-path outputs/attention_latent_injection_package_20260620t14471781966861z_8199dbc.zip` | pass, `full_rescue_applied_count=1`, `evidence_clean_fpr=0.0`, `evidence_attacked_fpr=0.03125` |
| `pytest tests/functional/test_geometric_rescue.py -q` | pass, 2 passed |
| `pytest -q` | pass, 72 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_12_threshold_calibration_metrics

| item | value |
| --- | --- |
| construction_unit_name | `threshold_calibration` |
| phase_status | `fixed_fpr_and_rescue_boundary_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/geometric_rescue/geometry_rescue_audit.json`; `outputs/geometric_rescue/aligned_detection_records.jsonl`; `outputs/aligned_rescoring/manifest.local.json` |
| expected_output_manifest | `outputs/threshold_calibration/manifest.local.json` |
| expected_outputs | `outputs/threshold_calibration/calibration_thresholds.json`; `outputs/threshold_calibration/fixed_fpr_operating_points.csv`; `outputs/threshold_calibration/standard_watermark_metrics.csv`; `outputs/threshold_calibration/quality_metrics_summary.csv`; `outputs/threshold_calibration/roc_curve_points.csv`; `outputs/threshold_calibration/det_curve_points.csv`; `outputs/threshold_calibration/score_distribution_table.csv`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/threshold_calibration/rescue_fpr_audit.csv`; `outputs/threshold_calibration/manifest.local.json` |
| blocking_items | fixed-FPR 与 rescue 的统计边界已经可审计并已冻结; 论文级完整方法 claim 仍为 `full_method_claim_ready=false`, 原因是当前样本规模、外部 baseline 正式结果与 dataset-level FID / KID 尚未补齐。 |
| fallback_path | 若后续重跑发现 calibration clean negative 的 observed FPR 超过目标 operating point, 应保持完整系统 fixed-FPR 主张为 unsupported; 不允许用 attacked negative 或 rescue 后样本改写 fixed-FPR 分母。 |
| invariants | fixed-FPR 分母仅使用 calibration clean negative; attacked negative 只作为鲁棒性诊断, 不治理 fixed-FPR 分母; rescue window 与 fail reason gate 已冻结, rescue 不改变 FPR 分母。 |
| next_stage_entry | 可继续进入真实攻击闭环、外部 baseline 导入、内部消融与论文产物审计; 若要形成论文级统计结论, 需要扩展到 full-main 样本规模并补齐 dataset-level FID / KID。 |

### stage12 当前完成内容

1. `threshold_degeneracy_report.json` 显示 `threshold_degenerate=false`, `fixed_fpr_boundary_ready=true`, `rescue_boundary_ready=true`, `fixed_fpr_and_rescue_boundary_ready=true`。
2. 当前 fixed-FPR 控制域为 `calibration_clean_negative`, 分母角色为 `clean_negative_only`, `attacked_negative_governs_fixed_fpr=false`。
3. 当前 rescue 控制域为 `evidence_clean_negative`, `rescue_changes_fpr_denominator=false`, `rescue_window_frozen=true`, `fail_reason_gate_frozen=true`。
4. 当前质量摘要已接收真实 aligned rescoring 的 pair-level PSNR、SSIM、MSE、MAE、LPIPS 与 CLIP score; FID / KID 仍属于 dataset-level 缺口, 不能由 pair-level 指标替代。
5. `supports_paper_claim=false` 与 `full_method_claim_ready=false` 是有意边界, 不是脚本失败。

### stage12 当前产物摘要

1. `target_fpr=0.05`, `calibration_negative_count=6`, `observed_fpr=0.0`, `allowed_false_positive_count=0`。
2. `calibrated_content_threshold=0.6343560311356602`, `threshold_value=0.6343560311356602`。
3. `attacked_fpr_diagnostic_exceeds_target=true`, 但 attacked negative 只用于鲁棒性诊断, 不进入 fixed-FPR 分母。
4. `real_aligned_rescore_count=3`, `perceptual_metrics_ready=true`, `input_attention_geometry_ready=true`, `input_image_quality_metrics_ready=true`。

### stage12 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_threshold_calibration_outputs.py` | pass, `fixed_fpr_boundary_ready=true`, `rescue_boundary_ready=true` |
| `pytest tests/functional/test_threshold_calibration.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## stage_13_attack_matrix_regeneration

| item | value |
| --- | --- |
| construction_unit_name | `attack_matrix` |
| phase_status | `real_attack_matrix_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/threshold_calibration/manifest.local.json`; `outputs/real_attack_evaluation/real_attack_manifest.local.json` |
| expected_output_manifest | `outputs/attack_matrix/manifest.local.json` |
| expected_outputs | `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/attacked_image_registry.jsonl`; `outputs/attack_matrix/attack_detection_records.jsonl`; `outputs/attack_matrix/attack_family_metrics.csv`; `outputs/attack_matrix/attack_strength_curve.csv`; `outputs/attack_matrix/score_retention_by_attack.csv`; `outputs/attack_matrix/rescue_by_attack.csv`; `outputs/attack_matrix/manifest.local.json`; `outputs/real_attack_evaluation/formal_attack_detection_records.jsonl` |
| blocking_items | 真实 attacked image 小样本闭环、再扩散类 GPU 验证与 formal attack detection 已完成并并入 attack matrix; 论文级 robustness 结论仍受外部 baseline 正式结果、full-main 样本规模和 dataset-level FID / KID 三项缺口阻断。 |
| fallback_path | 若后续真实攻击包缺失或 formal records 不可解析, attack matrix 应退回 `formal_attack_detection_ready=false` 并保留 unsupported reason, 不允许用 record-level proxy 冒充真实图像攻击结果。 |
| invariants | 攻击后检测复用 stage12 冻结的 fixed-FPR threshold、rescue window 和 fail reason gate; clean negative 与 attacked negative 分开统计; `supports_paper_claim=false`, `full_method_claim_ready=false`。 |
| next_stage_entry | 可继续推进外部 baseline 正式共同协议导入、内部消融和论文产物审计; 但不得把小样本真实攻击闭环写成论文级 full robustness 结论。 |

### stage13 当前完成内容

1. `outputs/real_attack_evaluation/attacked_images/` 中已有4张真实 attacked image, 覆盖 `img2img_regeneration`、`ddim_inversion_regeneration`、`sdedit_regeneration` 与 `diffusion_purification`。
2. `real_attacked_image_registry.jsonl` 已记录 source image digest、attacked image digest、攻击名称、图像路径和运行边界。
3. `formal_attack_detection_records.jsonl` 已接回正式 attack matrix schema, 并复用冻结的 fixed-FPR 与 rescue 边界重跑攻击后检测。
4. `outputs/attack_matrix/attack_manifest.json` 已记录 `real_attacked_image_closed_loop_ready=true`, `formal_attack_detection_ready=true`, `regeneration_attack_gpu_validation_ready=true`。
5. 当前真实攻击闭环属于小样本工程证据, 用于关闭“真实图像文件与再扩散路径是否可跑通”的工程缺口; 不等价于论文级 full-main robustness 统计。

### stage13 当前产物摘要

1. `attack_record_count=676`, `performed_attack_record_count=484`, `attack_metrics_ready=true`。
2. `formal_real_attack_record_count=4`, `real_attacked_image_count=4`, `measured_regeneration_attack_count=4`。
3. `gpu_attack_unsupported_count=0`, `gpu_attack_real_measurement_missing_count=0`, `regeneration_attack_status=real_gpu_formal_records_available`。
4. `real_attack_records_path=outputs/real_attack_evaluation/formal_attack_detection_records.jsonl`。

### stage13 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_attack_matrix_outputs.py` | pass, `formal_attack_detection_ready=true`, `real_attacked_image_count=4` |
| `pytest tests/functional/test_attack_matrix.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## stage_14_external_baseline_comparison

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_comparison` |
| phase_status | `baseline_protocol_and_formal_import_readiness_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/external_baseline_results/manifest.local.json`; `external_baseline/source_registry.json` |
| expected_output_manifest | `outputs/external_baseline_comparison/manifest.local.json` |
| expected_outputs | `outputs/external_baseline_results/baseline_result_records.jsonl`; `outputs/external_baseline_results/baseline_formal_import_readiness.csv`; `outputs/primary_baseline_formal_import/primary_baseline_formal_result_template.jsonl`; `outputs/primary_baseline_formal_import/primary_baseline_formal_template_coverage.csv`; `outputs/primary_baseline_formal_import/primary_baseline_formal_evidence_collection_plan.jsonl`; `outputs/primary_baseline_formal_import/primary_baseline_formal_evidence_collection_summary.json`; `outputs/external_baseline_comparison/baseline_observations.jsonl`; `outputs/external_baseline_comparison/baseline_result_records.jsonl`; `outputs/external_baseline_comparison/baseline_formal_import_validation_report.json`; `outputs/external_baseline_comparison/baseline_metrics.csv`; `outputs/external_baseline_comparison/baseline_comparison_table.csv`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| blocking_items | 8个 baseline 的 source registry 已可审计, 主表4个 baseline 已产生28条小样本候选记录, 但候选均未通过共同协议 validator; 因此 `baseline_results_ready=false`, `primary_baseline_formal_ready=false`, `supports_paper_claim=false`。 |
| fallback_path | 外部 baseline 无正式结果时只登记 `external_baseline_result_missing`; 允许保留小样本候选和拒绝原因, 但不允许把候选或 smoke 结果写入主表结论。 |
| invariants | baseline 与 SLM-WM 必须共享 prompt 协议、攻击矩阵协议和 fixed-FPR operating point; unsupported baseline 不进入论文主结论; 所有新增产物保持 `supports_paper_claim=false`。 |
| next_stage_entry | 可继续导入受治理正式结果或运行官方复现; 若要形成论文级外部 baseline 对比, 必须使主表 baseline 在 full-main prompt、fixed-FPR、攻击矩阵检测和证据路径四个边界上同时通过 validator。 |

### stage14 当前完成内容

1. `experiments/baselines/formal_import.py` 已提供主表 external baseline 正式导入 schema、候选记录 validator 和 per-baseline readiness 聚合。
2. `scripts/write_primary_baseline_result_candidates.py` 已写出 `baseline_result_records.jsonl`、`baseline_result_candidate_validation_report.json`、`baseline_formal_import_readiness.csv` 与 `baseline_formal_import_readiness_summary.json`。
3. `scripts/write_external_baseline_comparison_outputs.py` 已读取正式导入 readiness 摘要, 并将 `formal_result_ready_count`、`blocked_primary_baseline_ids` 和主要阻断原因透传到 `baseline_runtime_report.json`。
4. 当前主表候选来源为 Google Drive 中的小样本 GPU smoke 链路包; T2SMark 在缺少 full-main 包时也可从 GPU smoke observations 构造小样本候选, 候选可以审计, 但不能升级为正式论文结果。

### stage14 当前产物摘要

1. `baseline_runtime_report.json` 显示 `baseline_count=8`, `official_source_ready_count=8`, `baseline_observation_count=112`, `baseline_result_ready_count=0`, `baseline_results_ready=false`。
2. `formal_import_input_record_count=28`, `accepted_formal_import_count=0`, `rejected_formal_import_count=28`, `formal_import_issue_count=112`。
3. `formal_template_record_count=32`, `candidate_template_match_count=0`, `accepted_template_match_count=0`, `formal_template_coverage_ready_count=0`, `missing_candidate_template_count=32`, `missing_formal_template_count=32`, 说明当前候选尚未覆盖正式共同协议要求的 full-main 攻击模板。
4. `formal_evidence_collection_task_count=32`, `missing_formal_evidence_collection_task_count=32`, 说明后续真实 GPU 或受治理导入需要逐模板补齐正式证据记录。
5. `blocked_primary_baseline_ids=[tree_ring, gaussian_shading, shallow_diffuse, t2smark]`。
6. 主要阻断原因为 `attack_matrix_baseline_detection_ready_required`、`fixed_fpr_baseline_calibration_ready_required`、`full_main_prompt_protocol_ready_required` 和 `full_main_resource_profile_required`。

### stage14 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_primary_baseline_result_candidates.py --external-gpu-smoke-package-path <drive_zip>` | pass, `formal_import_candidate_record_count=28`, `accepted_formal_import_count=0` |
| `python scripts/write_primary_baseline_formal_import_protocol.py` | pass, `template_record_count=32`, `candidate_template_match_count=0`, `missing_formal_template_count=32`, `missing_formal_evidence_collection_task_count=32` |
| `python scripts/write_external_baseline_comparison_outputs.py` | pass, `baseline_results_ready=false`, `formal_result_ready_count=0` |
| `pytest tests/functional/test_primary_baseline_result_candidates.py tests/functional/test_external_baseline_comparison.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## stage_15_internal_ablation_evidence

| item | value |
| --- | --- |
| construction_unit_name | `internal_ablation_evidence` |
| phase_status | `ablation_protocol_ready_with_real_attack_inputs` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| expected_output_manifest | `outputs/internal_ablation_evidence/manifest.local.json` |
| expected_outputs | `outputs/internal_ablation_evidence/ablation_records.jsonl`; `outputs/internal_ablation_evidence/mechanism_ablation_table.csv`; `outputs/internal_ablation_evidence/method_pairwise_delta_table.csv`; `outputs/internal_ablation_evidence/ablation_by_attack_family.csv`; `outputs/internal_ablation_evidence/ablation_claim_summary.json`; `outputs/internal_ablation_evidence/manifest.local.json` |
| blocking_items | 内部消融协议已可重建并可读取最新 attack matrix; 论文级消融结论仍受 full-main 样本规模、外部 baseline 正式结果和 dataset-level FID / KID 缺口限制。 |
| fallback_path | 内部消融只用于机制必要性链路和表格重建链路; 若上游 attack matrix 或 threshold 边界退化, 消融 claim summary 必须保持 `supports_paper_claim=false`。 |
| invariants | 每个消融必须真实改变机制字段或判定边界; `full_slm_wm` 为参考行; `geo_direct_positive_audit` 只作为审计反例; 所有产物保持 `supports_paper_claim=false`。 |
| next_stage_entry | 可进入论文产物证据审计; 审计必须继续保留当前证据边界, 不得把小样本消融写成最终论文结论。 |

### stage15 当前完成内容

1. 内部消融产物已基于最新 attack matrix 重建, 上游真实再扩散攻击记录已经通过 attack matrix 进入消融输入边界。
2. `ablation_claim_summary.json` 仍保持 `supports_paper_claim=false`, 因为当前缺口来自论文级统计与外部 baseline, 而不是消融脚本不可运行。
3. 当前消融可用于检查机制路径、字段退化和表格重建, 不能替代 full-main 统计。

### stage15 当前产物摘要

1. `ablation_count=17`, `mechanism_group_count=7`, `ablation_protocol_ready=true`, `mechanism_coverage_ready=true`。
2. `external_baseline_result_ready=false`, 因此外部 superiority 与完整 robustness 主张不能成立。
3. `mechanism_ablation_table.csv`、`method_pairwise_delta_table.csv` 和 `ablation_by_attack_family.csv` 均可由 records 重建。

### stage15 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_internal_ablation_outputs.py` | pass |
| `pytest tests/functional/test_internal_ablation_evidence.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## stage_16_paper_artifact_evidence_audit

| item | value |
| --- | --- |
| construction_unit_name | `paper_artifact_evidence_audit` |
| phase_status | `evidence_gap_report_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/threshold_calibration/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/manifest.local.json`; `outputs/external_baseline_comparison/manifest.local.json`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/primary_baseline_small_sample_evidence/manifest.local.json`; `outputs/dataset_level_quality/manifest.local.json`; `outputs/internal_ablation_evidence/manifest.local.json` |
| expected_output_manifest | `outputs/paper_artifact_evidence_audit/manifest.local.json` |
| expected_outputs | `outputs/paper_artifact_evidence_audit/claim_audit_table.csv`; `outputs/paper_artifact_evidence_audit/paper_table_readiness.csv`; `outputs/paper_artifact_evidence_audit/paper_figure_readiness.csv`; `outputs/paper_artifact_evidence_audit/evidence_gap_list.csv`; `outputs/paper_artifact_evidence_audit/artifact_builder_readiness_report.json`; `outputs/paper_artifact_evidence_audit/evidence_audit_dry_run.json`; `outputs/paper_artifact_evidence_audit/submission_blocker_report.json`; `outputs/paper_artifact_evidence_audit/manifest.local.json` |
| blocking_items | `submission_ready=false`; 当前只剩3个主要缺口: `gap_baseline_results`, `gap_full_main_sample_scale`, `gap_dataset_level_fid_kid`。真实 attacked image 闭环、再扩散 GPU 验证、fixed-FPR 边界和 rescue 边界已不再列为当前阻断项。 |
| fallback_path | 当前只冻结 artifact builder 与 evidence audit 链路, 不冻结投稿结果; 不把预览表格、小样本候选或本地代理结果写成论文级结论。 |
| invariants | 所有新增产物保持 `supports_paper_claim=false`; 不手工补表; Notebook 不能直接写正式 records、tables、figures 或 reports; `main/` 不绑定外层运行目录。 |
| next_stage_entry | 需要先补齐外部 baseline 正式结果、full-main 样本规模统计和 dataset-level FID / KID, 再进入投稿冻结。 |

### stage16 当前完成内容

1. 论文产物证据审计已经消费最新 threshold、attack matrix、external baseline、小样本 baseline、dataset-level quality 和 internal ablation 产物。
2. `submission_blocker_report.json` 已将缺口收敛到3项, 并移除了已完成的真实 attacked image 与再扩散 GPU 验证阻断项。
3. `artifact_builder_readiness_report.json` 保持 artifact builder 可重建, 但 paper-ready artifact 数量仍为0。
4. 已新增 `paper_workflow/dataset_level_quality_run.ipynb` 与 `paper_workflow/colab_utils/dataset_level_quality.py`, 用于从 Google Drive 中的真实攻击包和 aligned rescoring 包生成 Inception 特征 JSONL, 再调用正式数据集级质量脚本重建 FID / KID 治理产物。

### stage16 当前产物摘要

1. `artifact_builder_ready=true`, `paper_artifact_audit_ready=true`, `claim_audit_row_count=9`, `table_readiness_row_count=7`, `figure_readiness_row_count=5`。
2. `rebuildable_artifact_count=11`, `blocked_artifact_count=1`, `paper_ready_artifact_count=0`。
3. `submission_blocker_report.json` 显示 `gap_count=3`, `critical_gap_count=2`, `blocking_claim_count=5`, `primary_blockers=[gap_baseline_results, gap_full_main_sample_scale, gap_dataset_level_fid_kid]`。
4. dataset-level quality 当前本地重建仍为 `dataset_level_quality_proxy_ready=true`, `formal_feature_backend_ready=false`, `formal_sample_scale_ready=false`, `formal_fid_kid_ready=false`; 新增 Colab 入口用于把 `formal_feature_backend_ready` 推进为可验证状态, 但小样本下 `formal_sample_scale_ready` 与 `formal_fid_kid_ready` 仍应保持 false。

### stage16 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_paper_artifact_evidence_audit_outputs.py` | pass, `gap_count=3`, `submission_ready=false` |
| `pytest tests/functional/test_dataset_level_quality.py tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 数据集级质量特征导入 helper 与 Colab 入口契约通过 |
| `pytest tests/functional/test_paper_artifact_evidence_audit.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## stage_17_pilot_full_submission_freeze

| item | value |
| --- | --- |
| construction_unit_name | `submission_readiness_gate` |
| phase_status | `blocked_by_current_evidence_gaps` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/paper_artifact_evidence_audit/manifest.local.json`; `outputs/paper_artifact_evidence_audit/artifact_builder_readiness_report.json`; `outputs/paper_artifact_evidence_audit/submission_blocker_report.json`; `outputs/paper_artifact_evidence_audit/evidence_gap_list.csv`; `outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json`; `docs/extraction_profiles.md`; `docs/release_boundary.md` |
| expected_output_manifest | `outputs/submission_readiness/submission_readiness_manifest.local.json` |
| expected_outputs | `outputs/submission_readiness/readiness_blocker_report.json`; `outputs/submission_readiness/required_evidence_inputs.csv`; `outputs/submission_readiness/release_profile_dry_run.csv`; `outputs/submission_readiness/submission_readiness_manifest.local.json` |
| blocking_items | `readiness_decision=blocked`; `submission_ready=false`; `required_input_count=3`; `critical_required_input_count=2`; `paper_ready_artifact_count=0`。 |
| fallback_path | 只生成投稿就绪阻断报告和 release dry-run 清单, 不导出投稿候选包, 不冻结论文级表格、图或 report。 |
| invariants | stage16 evidence audit 未通过投稿冻结前, release dry-run 可运行不等价于投稿就绪; 所有新增产物保持 `supports_paper_claim=false`; 不手工补表或手工标记 claim。 |
| next_stage_entry | 需要先补齐 `gap_baseline_results`、`gap_full_main_sample_scale` 和 `gap_dataset_level_fid_kid`, 然后重跑 stage16 与本门禁。当前不进行 TPR@FPR=0.01 或 TPR@FPR=0.001 的正式 full paper 运行。 |

### stage17 当前完成内容

1. `main/analysis/submission_readiness.py` 已将 stage16 证据审计产物、缺口列表、小样本 baseline 摘要和 release dry-run 摘要合成为投稿就绪门禁判定。
2. `scripts/write_submission_readiness_outputs.py` 已生成阻断报告、待补齐输入清单、release profile dry-run 表和 manifest。
3. 当前小样本 baseline 证据只允许解释为小样本共同协议边界, 不能支持正式 full paper 统计声明。
4. 最新小样本 baseline 摘要显示 `small_sample_evidence_ready=true`, `small_sample_common_protocol_ready=true`, `small_sample_baseline_covered_count=4`, 但 `small_sample_baseline_formal_import_ready_count=0`。

### stage17 当前产物摘要

1. `readiness_blocker_report.json` 显示 `readiness_decision=blocked`, `submission_ready=false`, `package_freeze_allowed=false`, `release_dry_run_ready=true`。
2. `required_evidence_inputs.csv` 只包含3个待补齐输入, 其中2个为 critical: 外部 baseline 正式结果与 full-main 样本规模; dataset-level FID / KID 为 major 缺口。
3. `paper_ready_artifact_count=0`, `formal_full_paper_run_requested=false`, `formal_full_paper_run_permitted=false`。
4. 已显式排除当前小样本流程下的 `tpr_at_fpr_0_01` 与 `tpr_at_fpr_0_001` 操作点。

### stage17 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_submission_readiness_outputs.py` | pass, `required_input_count=3`, `critical_required_input_count=2` |
| `pytest tests/functional/test_submission_readiness.py -q` | pass |
| `python tools/harness/run_all_audits.py` | pass |

## real_gpu_aligned_rescoring_workflow

| item | value |
| --- | --- |
| construction_unit_name | `aligned_rescoring` |
| phase_status | `colab_pair_metric_workflow_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attention_geometry/manifest.local.json`; `outputs/content_carriers/manifest.local.json`; `outputs/attention_latent_update/manifest.local.json` |
| expected_output_manifest | `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json` |
| expected_outputs | `paper_workflow/aligned_rescoring_run.ipynb`; `paper_workflow/colab_utils/aligned_rescoring.py`; `outputs/aligned_rescoring/aligned_rescoring_records.jsonl`; `outputs/aligned_rescoring/aligned_rescoring_result.json`; `outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv`; `outputs/aligned_rescoring/aligned_rescoring_environment_report.json`; `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json`; `GoogleDrive/SLM/aligned_rescoring/aligned_rescoring_package_<utc>_<short_commit>.zip` |
| blocking_items | 鏈湴鐜鏃?GPU 鍜岀湡瀹?SD3.5 Medium 鏉冮噸, 鍥犳鏈瀹屾垚 Colab workflow 涓?repository helper 鐨?LPIPS / CLIP pair-level 鎸囨爣鎺ュ叆; LPIPS / CLIP 榛樿鍦?CPU 涓婅绠椾互閬垮紑 SD3.5 pipeline 鍗犵敤 GPU 鍚庣殑鏄惧瓨鍘嬪姏, 鏂扮湡瀹炰骇鐗╅渶瑕佸湪 Colab GPU 涓噸鏂拌繍琛?notebook 鍚庡洖浼犲璁°€?|
| fallback_path | 鑻ユ病鏈?ready attention geometry 鍖呫€丠F_TOKEN銆丟PU runtime銆佺湡瀹?latent callback 鎴?required pair-level perceptual metrics, helper 浼氬啓鍑?fail result 鍜?unsupported reason, 涓嶄細浼€?real aligned score 鎴栨劅鐭ユ寚鏍囥€?|
| invariants | Notebook 鍙綔涓哄叆鍙? 姝ｅ紡閫昏緫浣嶄簬 `paper_workflow/colab_utils/aligned_rescoring.py`; 杈撳嚭浠嶄繚鎸?`supports_paper_claim=false` 鍜?`full_method_claim_ready=false`, 鐩村埌閲嶆柊杩愯 geometric rescue 涓?threshold calibration 骞跺璁?FPR銆?|
| next_stage_entry | Colab 鐢熸垚骞跺洖浼?aligned rescoring 鍖呭悗, 鏈湴搴斿厛瀹¤鍖呭唴 records銆乹uality metrics銆乵anifest 鍜?environment report, 鍐嶅喅瀹氭槸鍚﹂噸璺?geometric rescue銆乼hreshold calibration 涓?attack matrix銆?|

### aligned rescoring workflow 宸插畬鎴愬唴瀹?

1. 鏂板 `paper_workflow/colab_utils/aligned_rescoring.py`, 鏀寔璇诲彇 ready attention geometry 鍖呫€侀噸寤?prompt / semantic / content / attention update 杈撳叆閾? 閫夋嫨 active attention carrier, 鍦ㄧ湡瀹?SD3.5 Medium latent callback 涓幏鍙栧榻愬墠鍚?latent 鎶曞奖骞堕噸鏂拌绠?LF/HF 鍐呭鍒嗘暟銆?
2. 鏂板骞舵洿鏂?`paper_workflow/aligned_rescoring_run.ipynb`, 鏀寔 Colab 鍐峰惎鍔? 鎸傝浇 Google Drive銆佸畨瑁呭綋鍓?Colab 鍙繍琛屼緷璧栫粍鍚堝拰 LPIPS 鍙€変緷璧栥€佹媺鍙栦粨搴撱€佽鍙?`HF_TOKEN`銆佹鏌?GPU銆佹墽琛岀湡瀹?aligned rescoring, 璁＄畻 LPIPS 涓?CLIP pair-level 鎸囨爣, 骞跺皢缁撴灉鍖呬繚瀛樺埌 `GoogleDrive/SLM/aligned_rescoring/`銆?
3. 鏂板鎵撳寘鍑芥暟 `package_aligned_rescoring_outputs`, 浼氭妸 aligned rescoring records銆乺esult銆乹uality metrics銆乪nvironment report銆乵anifest銆乤ttention update 鏂规硶鏂囦欢鍜?package input manifest 绾冲叆 zip銆?
4. 鏇存柊 `tests/constraints/test_notebook_entrypoint_contract.py`, 瑕嗙洊鏂?Notebook 鍏ュ彛濮旀墭銆佹棤鎵ц杈撳嚭銆丏rive 闀滃儚璺緞鍜屾墦鍖呬骇鐗╂牳瀵广€?
5. 鏇存柊 `docs/field_registry.md`, 鐧昏鐪熷疄 aligned rescoring銆乴atent projection銆丩PIPS / FID / KID / CLIP 鐘舵€併€乧lean / aligned CLIP score銆丆LIP delta 鍜岃川閲忔寚鏍囩浉鍏冲瓧娈点€?
6. 鏂板杞婚噺娴嬭瘯 `tests/functional/test_aligned_rescoring_metrics.py`, 楠岃瘉 LPIPS / CLIP pair-level ready 杈圭晫銆侀粯璁ら厤缃拰璐ㄩ噺鎸囨爣琛ㄥ瓧娈点€?
7. 鏇存柊鎰熺煡鎸囨爣璇婃柇: 鑻?LPIPS 鎴?CLIP 鏈?measured, `unsupported_reason` 浼氬啓鍏?`lpips_status` 涓?`clip_score_status`, 璐ㄩ噺琛ㄤ細璁板綍瀵瑰簲 error type 鍜屽帇缂╅敊璇俊鎭? Notebook 鍦ㄦ柇瑷€澶辫触鍓嶄細鎵撳嵃璐ㄩ噺琛ㄤ究浜庡畾浣嶃€?
8. 鏇存柊 CLIP 璁＄畻鍏煎璺緞: 浼樺厛浣跨敤 `get_image_features` / `get_text_features`, 鑻ュ綋鍓?transformers 鐗堟湰缂哄皯璇?API, 鍒欓€€鍥炲埌 `CLIPModel` forward 杈撳嚭涓殑 `image_embeds` / `text_embeds` 鎴?`logits_per_image`銆?

### aligned rescoring workflow 褰撳墠杈圭晫

1. 褰撳墠 workflow 榛樿鍙繍琛屽皯閲?active attention carrier, 鐢ㄤ簬楠岃瘉鐪熷疄 GPU latent 鎶曞奖閲嶆墦鍒嗛摼璺? 涓嶆槸 full-main 瑙勬ā缁熻銆?
2. `aligned_rescoring_quality_metrics.csv` 榛樿璁板綍 PSNR銆丼SIM銆丮SE銆丮AE銆丩PIPS銆乣clip_score_clean`銆乣clip_score_aligned` 鍜?`clip_score_delta`; 鑻?LPIPS 鎴?CLIP 璁＄畻澶辫触涓?`require_pair_perceptual_metrics=true`, 鏈繍琛岀殑 `run_decision` 搴斾负 `fail`, 骞跺湪 `unsupported_reason` 涓庤川閲忚〃涓繚鐣欒瘖鏂姸鎬併€?
3. FID / KID 浠嶆槸 dataset-level metric, 褰撳墠 pair-level Colab workflow 涓嶈绠?FID / KID, 缁х画鍐欏叆鏄庣‘鐨?unsupported status銆?
4. 鏂扮湡瀹?aligned rescoring 鍖呭洖浼犲悗, 蹇呴』閲嶆柊瀹¤ `real_aligned_rescore_count > 0`銆乣image_quality_metrics_ready=true`銆乣perceptual_metrics_ready=true`銆佺幆澧冧緷璧栫増鏈拰鎵€鏈夎緭鍏?manifest, 涔嬪悗鎵嶈兘閲嶈窇 fixed-FPR 鐩稿叧浜х墿銆?

### aligned rescoring result 涓嬫父浼犳挱璁板綍

1. 宸插皢 `outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip` 浣滀负闃堝€兼牎鍑嗙殑鏄惧紡杈撳叆, 骞跺湪 `outputs/threshold_calibration/manifest.local.json` 涓褰曡緭鍏ヨ矾寰勪笌 SHA256 鎽樿 `ac1c8578f611de53aaae68ab22ecc667746090272bcb5c95d2e7844b6913964e`銆?
2. `outputs/threshold_calibration/quality_metrics_summary.csv` 宸叉敼涓轰紭鍏堜娇鐢?aligned rescoring 鍖呬腑鐨勭湡瀹?pair-level 璐ㄩ噺鎸囨爣; PSNR銆丼SIM銆丮SE銆丮AE銆丩PIPS 涓?CLIP score 鍧囦负 measured, FID / KID 淇濇寔 dataset-level unsupported銆?
3. `outputs/threshold_calibration/threshold_degeneracy_report.json`銆乣outputs/threshold_calibration/manifest.local.json`銆乣outputs/attack_matrix/attack_manifest.json` 涓?`outputs/attack_matrix/manifest.local.json` 鍧囧凡鍐欏叆 `aligned_rescoring_quality_metrics_ready=true`銆乣perceptual_metrics_ready=true`銆乣aligned_rescoring_record_count=3` 涓?`real_aligned_rescore_count=3`銆?
4. 璇ヤ紶鎾彧瑙ｅ喅鐪熷疄 aligned rescoring 鍖呯殑璐ㄩ噺鎸囨爣涓?provenance 杩涘叆涓嬫父浜х墿鐨勯棶棰? fixed-FPR 缁熻浠嶆部鐢?governed geometric rescue records, 鍥犳 `evidence_attacked_fpr=0.15625` 涓?`full_method_claim_ready=false` 涓嶅洜鏈浼犳挱鑰屾敼鍙樸€?

### aligned rescoring workflow 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `pytest tests/functional/test_aligned_rescoring_metrics.py tests/constraints/test_notebook_entrypoint_contract.py -q` | pass, 18 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## real_gpu_attack_evaluation_workflow

| item | value |
| --- | --- |
| construction_unit_name | `real_attack_evaluation` |
| phase_status | `real_gpu_small_sample_attack_closed_loop_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json`; `outputs/attack_matrix/manifest.local.json`; `outputs/threshold_calibration/manifest.local.json` |
| expected_output_manifest | `outputs/real_attack_evaluation/real_attack_manifest.local.json` |
| expected_outputs | `paper_workflow/real_attack_evaluation_run.ipynb`; `paper_workflow/colab_utils/real_attack_evaluation.py`; `outputs/real_attack_evaluation/attacked_images/*.png`; `outputs/real_attack_evaluation/real_attack_detection_records.jsonl`; `outputs/real_attack_evaluation/formal_attack_detection_records.jsonl`; `outputs/real_attack_evaluation/real_attacked_image_registry.jsonl`; `outputs/real_attack_evaluation/real_attack_family_metrics.csv`; `outputs/real_attack_evaluation/real_attack_environment_report.json`; `outputs/real_attack_evaluation/real_attack_manifest.local.json`; `GoogleDrive/SLM/real_attack_evaluation/real_attack_evaluation_package_<utc>_<short_commit>.zip` |
| blocking_items | 小样本真实 GPU 链路已跑通; 当前不再阻断 attack matrix 重建。论文级 robustness 仍需要 full-main 样本规模和后续 evidence audit。 |
| fallback_path | 若后续 Colab 结果包缺失、模型不可用或 DDIM inversion 后端失败, helper 必须写出 `run_decision=fail` 与 `unsupported_reason`, 不得伪造 attacked image 或 formal detection records。 |
| invariants | Notebook 只作为远程入口; 正式逻辑位于 `paper_workflow/colab_utils/real_attack_evaluation.py`; 产物保持 `supports_paper_claim=false`, 直到 full-main 统计和证据审计通过。 |
| next_stage_entry | 真实攻击包已回传并通过本地审计后, 应继续把 formal records 向 attack matrix、paper artifact evidence audit 和 submission readiness 传播。 |

### real attack evaluation workflow 当前完成内容

1. Colab GPU 运行已生成4个再扩散类真实攻击结果, 覆盖 img2img、DDIM inversion、SDEdit 与 diffusion purification。
2. 每条真实攻击记录均已写出 source image digest、attacked image digest、攻击名称、图像路径和 formal detection 字段。
3. `run_decision=pass`, `real_attack_record_count=4`, `real_attacked_image_count=4`, `formal_attack_detection_ready=true`。
4. 运行环境已记录 Colab L4、CUDA 12.8、Python 3.12.13、torch 2.11.0+cu128、diffusers 0.38.0 和 transformers 5.12.1。

### real attack evaluation workflow 当前边界

1. 该 workflow 关闭的是“小样本真实图像攻击闭环能否跑通”的工程缺口, 不是论文级 robustness 结论。
2. formal fixed-FPR 边界沿用 threshold calibration: clean negative 控制 FPR, attacked negative 作为鲁棒性诊断。
3. 后续 full-main 运行可以复用该 workflow, 但必须重新生成更大样本量的 records、表格和审计报告。

## external_baseline_governed_ingestion

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_comparison` |
| phase_status | `formal_import_readiness_audit_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-23` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/external_baseline_results/manifest.local.json`; `external_baseline/source_registry.json` |
| expected_output_manifest | `outputs/external_baseline_comparison/manifest.local.json` |
| expected_outputs | `outputs/external_baseline_results/baseline_result_records.jsonl`; `outputs/external_baseline_results/baseline_result_candidate_validation_report.json`; `outputs/external_baseline_results/baseline_formal_import_readiness.csv`; `outputs/external_baseline_results/baseline_formal_import_readiness_summary.json`; `outputs/primary_baseline_formal_import/primary_baseline_formal_result_template.jsonl`; `outputs/primary_baseline_formal_import/primary_baseline_formal_template_coverage.csv`; `outputs/primary_baseline_formal_import/primary_baseline_formal_template_coverage_summary.json`; `outputs/primary_baseline_formal_import/primary_baseline_formal_evidence_collection_plan.jsonl`; `outputs/primary_baseline_formal_import/primary_baseline_formal_evidence_collection_summary.json`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| blocking_items | 当前4条主表候选均未通过正式导入 validator; `accepted_formal_import_count=0`, `formal_result_ready_count=0`, `primary_baseline_formal_ready=false`。 |
| fallback_path | 允许保存候选记录与拒绝原因作为后续补证输入; 不允许把小样本候选、GPU smoke observation 或 legacy reference 结果直接手工写入主表。 |
| invariants | 第三方源码缓存仍由 `external_baseline/source_registry.json` 记录; 本项目提交 adapter、schema、导入报告与测试, 不把不受治理的第三方输出当作 supported claim。 |

### external baseline 正式导入推进内容

1. 已建立主表 external baseline 正式共同协议导入 readiness 表, 对 Tree-Ring、Gaussian Shading、Shallow Diffuse 与 T2SMark 分别聚合候选数量、接受数量、拒绝数量和阻断原因。
2. 已把 readiness 摘要并入 `baseline_runtime_report.json`, 使下游审计能够直接读取 `blocked_primary_baseline_ids` 与 `dominant_formal_import_blocking_reasons`。
3. 已新增正式模板覆盖检查, 将 full-main 攻击模板覆盖情况写入 `primary_baseline_formal_template_coverage.csv` 与 `primary_baseline_formal_template_coverage_summary.json`。
4. 已新增正式证据收集计划, 将缺失 full-main 模板转换为逐项补证任务, 写入 `primary_baseline_formal_evidence_collection_plan.jsonl` 与 `primary_baseline_formal_evidence_collection_summary.json`。
5. 已补齐 method-faithful SD3.5 adapter 的图像级攻击覆盖入口, 默认覆盖 `jpeg_compression`、`gaussian_noise`、`gaussian_blur`、`rotation`、`resize`、`crop`、`crop_resize` 和 `composite_geometric_attacks`, 并记录 attacked image provenance。
6. 已允许 `write_primary_baseline_result_candidates.py` 在 T2SMark full-main 包缺失时, 从 external GPU smoke 包中的 T2SMark observations 构造小样本候选记录, 从而保持4个主表 baseline 的小样本证据边界完整。
7. 已修正小样本证据摘要的 common protocol readiness 聚合方式: 多攻击记录按 baseline 覆盖判断, 不再因单个 baseline 产生多条攻击记录而误判 common protocol 未就绪。
8. 当前官方源码缓存登记显示8个 baseline 的源码入口可检查, 但正式结果仍为未就绪。
9. 下一步应在共同协议下补齐 full-main prompt、fixed-FPR baseline calibration、attack matrix baseline detection 和正式证据路径, 再重新运行导入 validator。

### external baseline 当前产物摘要

1. `baseline_result_candidate_summary.json` 显示 `formal_import_candidate_record_count=28`, `accepted_formal_import_count=0`, `rejected_formal_import_count=28`, `formal_import_issue_count=112`。
2. `baseline_formal_import_readiness.csv` 对4个主表 baseline 均给出 `formal_result_ready=false`。
3. `baseline_formal_import_readiness_summary.json` 显示 `blocked_primary_baseline_ids=[tree_ring, gaussian_shading, shallow_diffuse, t2smark]`。
4. `primary_baseline_formal_template_coverage_summary.json` 显示 `formal_template_record_count=32`, `candidate_template_match_count=0`, `accepted_template_match_count=0`, `missing_candidate_template_count=32`, `missing_formal_template_count=32`。
5. `primary_baseline_formal_evidence_collection_summary.json` 显示 `formal_evidence_collection_task_count=32`, `missing_formal_evidence_collection_task_count=32`。
6. `baseline_runtime_report.json` 显示 `official_source_ready_count=8`, `formal_import_input_record_count=28`, `baseline_results_ready=false`, `supports_paper_claim=false`。
7. `primary_baseline_small_sample_evidence_summary.json` 显示 `small_sample_evidence_ready=true`, `small_sample_common_protocol_ready=true`, `covered_primary_baseline_count=4`, 但 `formal_import_ready_count=0` 且 `supports_paper_claim=false`。

### external baseline 当前验证结果

| command | result |
| --- | --- |
| `python scripts/write_primary_baseline_result_candidates.py --external-gpu-smoke-package-path outputs/external_baseline_gpu_smoke_package_20260623t14351782225358z_020d16f.zip` | pass, `formal_import_candidate_record_count=28` |
| `python scripts/write_primary_baseline_formal_import_protocol.py` | pass, `template_record_count=32`, `candidate_template_match_count=0`, `missing_formal_template_count=32`, `missing_formal_evidence_collection_task_count=32` |
| `python scripts/write_external_baseline_comparison_outputs.py` | pass |
| `python scripts/write_primary_baseline_small_sample_evidence_outputs.py` | pass, `small_sample_evidence_ready=true`, `small_sample_common_protocol_ready=true` |
| `pytest tests/functional/test_primary_baseline_result_candidates.py tests/functional/test_primary_baseline_small_sample_evidence.py tests/functional/test_primary_baseline_formal_import.py tests/functional/test_external_baseline_comparison.py -q` | pass |

## primary_baseline_reproduction_protocol

| item | value |
| --- | --- |
| construction_unit_name | `primary_baseline_reproduction` |
| phase_status | `official_execution_plan_ready_result_import_required` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `external_baseline/source_registry.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/attack_family_metrics.csv` |
| expected_output_manifest | `outputs/primary_baseline_reproduction/manifest.local.json` |
| expected_outputs | `outputs/primary_baseline_reproduction/primary_baseline_execution_plan.jsonl`; `outputs/primary_baseline_reproduction/primary_baseline_result_record_template.jsonl`; `outputs/primary_baseline_reproduction/primary_baseline_reproduction_report.json`; `outputs/primary_baseline_reproduction/manifest.local.json` |
| blocking_items | 当前只冻结官方复现命令、依赖画像和共同协议结果模板, 尚未运行真实 GPU baseline 复现, 因此 `baseline_results_ready=false`。 |
| fallback_path | 若官方代码无法在统一环境运行, 应使用隔离环境或受治理导入记录进入 `outputs/external_baseline_results/baseline_result_records.jsonl`, 不得手工填表。 |
| invariants | 第三方源码仍由 `external_baseline/` 缓存且不提交; 主表 baseline 结果必须通过共同协议键进入 records, 再由脚本重建对比表。 |

### primary baseline 推进内容

1. 新增 `experiments/baselines/primary_reproduction.py`, 冻结 Tree-Ring、Gaussian Shading、Shallow Diffuse 和 T2SMark 的官方入口命令、依赖画像、模型对齐状态和结果适配器名称。
2. 新增 `scripts/write_primary_baseline_reproduction_plan.py`, 从源码登记文件与攻击矩阵读取输入, 写出主表 baseline 官方复现计划和共同协议结果导入模板。
3. 新增 `tests/functional/test_primary_baseline_reproduction_plan.py`, 验证 4 个主表 baseline 均进入计划, T2SMark 被标记为 SD3.5 Medium 原生入口, 其他旧版 SD 系 baseline 标记为需要协议适配。
4. 当前计划将 Tree-Ring、Gaussian Shading、Shallow Diffuse 归入 `legacy_stable_diffusion_requires_protocol_adapter`, 将 T2SMark 归入 `sd35_medium_native_entrypoint`。

### primary baseline 当前边界

1. 本次完成的是主表 baseline 复现协议和结果导入模板, 不是外部 baseline 真实指标复现。
2. 真实复现应在隔离 GPU 环境中运行官方代码, 并把结果转换成 `baseline_result_records.jsonl` 后再重建 `external_baseline_comparison`。
3. 当前所有新增产物仍保持 `supports_paper_claim=false`, 不能支持 baseline superiority 结论。

### primary baseline 验证结果

| command | result |
| --- | --- |
| `python -m py_compile experiments/baselines/primary_reproduction.py scripts/write_primary_baseline_reproduction_plan.py tests/functional/test_primary_baseline_reproduction_plan.py` | pass |
| `python scripts/write_primary_baseline_reproduction_plan.py` | pass, `primary_baseline_count=4`, `result_record_template_count=56` |
| `pytest -q` | pass, 109 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |



## external_baseline_adapter_command_evidence_protocol

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_adapter_command_evidence` |
| phase_status | `adapter_command_observation_evidence_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `external_baseline/source_registry.json` |
| expected_output_manifest | `outputs/external_baseline_execution/baseline_execution_manifest.json` |
| expected_outputs | `outputs/external_baseline_command_plan/baseline_command_plan.json`; `outputs/external_baseline_command_plan/baseline_command_plan_manifest.json`; `outputs/external_baseline_execution/baseline_command_results.json`; `outputs/external_baseline_execution/baseline_observations.json`; `outputs/external_baseline_execution/baseline_execution_manifest.json` |
| blocking_items | Tree-Ring、Gaussian Shading 和 Shallow Diffuse 已具备 SD3.5 latent 级 GPU smoke adapter, 但仍需要官方完整复现或受治理结果导入才能支撑论文级外部 baseline 对比。 |
| fallback_path | 若官方源码暂不能直接适配 SD3.5 Medium, 只能保留 `contract-only` 诊断或导入受治理结果, 不得声明论文级 baseline 结论。 |
| invariants | 官方源码快照位于 `external_baseline/*/*/source/` 且不由 git 跟踪; 项目维护 adapter、命令计划脚本、执行脚本和证据校验脚本必须接受 harness 审计。 |

### external baseline 并入方法修正

1. 根 `.gitignore` 不再忽略整个 `external_baseline/`, 改由 `external_baseline/.gitignore` 仅忽略第三方 `source/` 与 adapter 临时 `artifacts/` 子树。
2. `tools/harness/lib/file_scanner.py` 已改为扫描 `external_baseline/` 中的 adapter、README 和登记文件, 但跳过第三方源码快照。
3. 新增 `experiments/baselines/command_adapter.py`、`command_plan.py`、`observation_io.py` 和 `evidence_validator.py`, 形成 command plan、execution、observation 和 evidence 的统一入口。
4. 新增 `scripts/build_external_baseline_command_plan.py`、`scripts/run_external_baseline_command_plan.py` 和 `scripts/validate_external_baseline_evidence.py`, 所有仓库命令输出默认写入 `outputs/`。
5. 主表 baseline 已新增项目维护 adapter 路径。T2SMark adapter 可读取官方 `results.json`; Tree-Ring、Gaussian Shading 和 Shallow Diffuse 当前提供 SD3.5 latent 级 GPU smoke adapter, 正式指标仍需补齐官方完整复现或受治理导入路径。
6. `external_baseline/source_registry.json` 已补充 `adapter_path`、`adapter_status`、`model_alignment_status` 和 `official_source_tracked` 字段。

### 当前边界

1. 本次变更建立外部 baseline 的实施流程, 不伪造外部 baseline 真实指标。
2. `contract-only` 只证明命令编排、adapter 落盘和 harness 边界可用。
3. 论文级对比必须有 `formal_result_claim=true`、真实 `evidence_paths`、官方源码 commit、运行日志和可重建 observation 或受治理结果记录。


## external_baseline_gpu_smoke_colab_entrypoint

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_gpu_smoke` |
| phase_status | `colab_gpu_smoke_entrypoint_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `external_baseline/source_registry.json`; Google Drive 历史 `external_baseline_gpu_smoke_package_*.zip` 可选 |
| expected_output_manifest | `outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_manifest.local.json` |
| expected_outputs | `outputs/external_baseline_gpu_smoke/t2smark_official/**`; `outputs/external_baseline_gpu_smoke/execution/baseline_observations.json`; `outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_summary.json`; Google Drive `SLM/external_baseline_gpu_smoke/external_baseline_gpu_smoke_package_<utc>_<short_commit>.zip` |
| blocking_items | 该 Notebook 已覆盖 T2SMark SD3.5 Medium 最小真实 GPU smoke, 并把 Tree-Ring、Gaussian Shading、Shallow Diffuse 接入 SD3.5 method-faithful GPU smoke adapter; 默认样本量为5, 默认覆盖8类图像级攻击, 仍不支持论文级外部 baseline 对比结论。 |
| fallback_path | 若 Google Drive 中已有可复用官方结果包, helper 会先解包复用; 若缺失, 则按源码登记表下载 T2SMark 官方源码并重新生成官方 `results.json`。 |
| invariants | Notebook 只负责远程入口和打包, 真实逻辑位于 `paper_workflow/colab_utils/external_baseline_gpu_smoke.py`; 所有持久输出写入 `outputs/` 并镜像到 Google Drive。 |

### external baseline GPU smoke 入口内容

1. 新增 `paper_workflow/external_baseline_gpu_smoke_run.ipynb`, 支持 Colab 冷启动挂载 Google Drive、拉取仓库、读取 `HF_TOKEN`、检查 CUDA、执行最小 T2SMark SD3.5 Medium 真实 GPU smoke, 并在同一命令计划中运行四个主表 external baseline adapter。
2. 新增 `paper_workflow/colab_utils/external_baseline_gpu_smoke.py`, 将历史包复用、官方源码缓存补齐、官方 `results.json` 生成、image pair 构造、主表 baseline adapter 命令计划执行和 zip 打包收敛到 helper。
3. 前序结果判断边界为: 优先查找 Google Drive `external_baseline_gpu_smoke_package_*.zip`, 仅解出 `outputs/external_baseline_gpu_smoke/` 下可复用文件; 若 `results.json` 存在且允许复用, 不重新运行官方推理; 否则执行真实 GPU 生成。
4. 当前产物显式设置 `supports_paper_claim=false`, 只能证明外部 baseline 链路可运行, 不能替代 full-main prompt split、样本量冻结、固定 FPR 与 baseline 主表统计。
5. Tree-Ring、Gaussian Shading 和 Shallow Diffuse 当前采用项目治理内的 SD3.5 method-faithful smoke adapter: 它验证 16-channel latent 形状、GPU 张量路径、clean / positive observation 输出、8类图像级攻击 observation 输出和 manifest 边界, 不等同于第三方官方完整复现。
6. `external_baseline_gpu_smoke_package_20260623t14351782225358z_020d16f.zip` 已显示 `primary_baseline_attacked_image_count=240`, 三个 method-faithful baseline 各80张 attacked image, source / attacked image digest 均可核验。


### external baseline GPU smoke Colab 兼容修正

1. Colab 返回包显示 T2SMark 官方 `src/inversion/inverse_diffusion3.py` 在 Diffusers 0.38.0 / Transformers 5.12.1 环境中因 `Union` 未定义而在类定义阶段失败。
2. 修正策略是在项目 helper 中对第三方源码缓存应用最小兼容补丁, 显式补齐 `torch`、`typing` 与 `PipelineImageInput` 导入; 第三方 `source/` 子树仍不进入 git 提交。
3. 前序结果复用边界同步修正: 若 Google Drive 历史包中已有可复用 `results.json`, helper 会跳过源码缓存准备, 避免在不需要重新推理时被 GitHub 源码下载或补丁流程阻断。
4. 该修正只提升 cold-start smoke 链路鲁棒性, 不改变 `supports_paper_claim=false` 的证据边界。


### external baseline GPU smoke provenance 修正

1. 已发现历史失败包复用时 `t2smark_image_pairs.json` 可能保留空的 `generated_image_path` 与 `generated_image_digest`, 但新的官方图像已经生成。
2. 修正策略是将 image pair 构造改为以当前 `t2smark_official/.../images/` 目录为准: 若已有 image pair 与当前图像路径或 digest 不一致, helper 会自动重写 `t2smark_image_pairs.json`。
3. 重新生成或刷新方式为重新运行 `paper_workflow/external_baseline_gpu_smoke_run.ipynb`; helper 会在官方推理或结果复用后自动执行刷新, 不需要手工编辑 JSON。


### external baseline 主表证据边界推进

1. 新增 `experiments/baselines/primary_evidence.py`, 将四个主表 baseline 的 adapter smoke 链路状态与正式共同协议结果边界分开记录。
2. 新增 `scripts/write_primary_baseline_evidence_outputs.py`, 可读取 `external_baseline_gpu_smoke` 的 command results 与 observations, 也可直接读取 GPU smoke zip 包, 写出 `outputs/primary_baseline_evidence/primary_baseline_evidence_records.jsonl`、summary 和 manifest。
3. 该证据边界明确记录: Tree-Ring、Gaussian Shading 和 Shallow Diffuse 虽已具备 SD3.5 latent smoke 链路, 但仍缺方法忠实 SD3.5 adapter、full-main prompt 协议、fixed-FPR 校准、攻击矩阵检测和正式证据路径。
4. 该推进不改变 `supports_paper_claim=false`; 作用是防止 smoke observation 被误升级为论文级主表 external baseline 指标。



### 主表 baseline 正式导入协议与 T2SMark full-main 路径补充

1. 新增 `experiments/baselines/formal_import.py`, 将主表 external baseline 的正式结果导入边界集中到 schema validator 中, 下游 `external_baseline_comparison` 只消费 `accepted_records`, 不再把 GPU smoke observation 或缺少 fixed-FPR / full-main / attack matrix 边界的记录纳入正式比较。
2. 新增 `scripts/write_primary_baseline_formal_import_protocol.py`, 可写出正式导入 schema、主表结果模板、正式模板覆盖、证据收集计划、候选记录校验报告和 manifest。该脚本只生成治理产物, 不手工填充论文结果。
3. 新增 `paper_workflow/colab_utils/t2smark_full_main_reproduction.py` 与 `paper_workflow/t2smark_full_main_reproduction_run.ipynb`, 支持 Colab 冷启动下读取 `configs/paper_main_full_prompts.txt`, 运行 T2SMark SD3.5 Medium full-main 官方入口, 生成 image_pairs、统一 adapter observations、正式导入候选记录、validator 报告, 并打包镜像到 Google Drive。
4. 当前 T2SMark full-main 路径默认 `supports_paper_claim=false`。若 fixed-FPR 校准和攻击矩阵检测未闭合, validator 会保留 `formal_import_validation_ready=false`, 防止 raw full-main 官方结果被误声明为论文级主表 robustness 结论。

### dataset-level quality 打包自描述修正

1. Google Drive 审计边界以 `SLM/dataset_level_quality/` 为准, 本地 `outputs/` 中同名 zip 仅视为手动下载副本。
2. 数据集级质量打包逻辑已改为只从真实攻击包解出 `real_attacked_image_registry.jsonl`; source 与 attacked image 均通过 `outputs/dataset_level_quality/materialized_image_inputs/` 物化后进入结果包, 避免包内记录指向未打包的前序目录图像。
3. zip 内部的 `dataset_level_quality_archive_summary.json` 不再写入空的最终 `archive_digest` 或 `drive_archive_digest`; 内部摘要改用 `archive_payload_digest` 表示打包输入条目的稳定摘要。
4. 最终 zip 文件 SHA-256 与 Google Drive 镜像 SHA-256 继续写入同目录 sidecar summary 与 manifest, 通过 `archive_digest_scope` 和 `final_archive_digest_available_in_sidecar` 明确摘要边界。

### external baseline comparison 小样本边界联动

1. `external_baseline_comparison` 已补充读取主表 baseline 小样本证据摘要, 在 runtime report 中同时暴露正式导入状态与小样本共同协议边界。
2. 小样本字段只用于审计可见性, 不改变 `baseline_results_ready=false` 与 `supports_paper_claim=false` 的正式论文声明边界。
3. 当 `primary_baseline_small_sample_evidence_summary.json` 缺失时, comparison 仍可重建正式导入状态, 但会显式记录小样本 baseline 证据未就绪。
4. 该联动用于把已完成的小样本主表 baseline 链路向下游 evidence audit / submission readiness 传播, 不触发正式 full paper 样本规模、TPR@FPR=0.01 或 TPR@FPR=0.001 运行。

### external baseline formal evidence path resolution

1. `external_baseline_comparison` 已新增 `baseline_formal_evidence_path_resolution_report.json`, 用于单独汇总正式导入候选记录的 evidence paths 在当前工作区或挂载目录下是否可解析。
2. 该报告解释 candidate validator 与 comparison 重新校验之间可能出现的差异: 若本地 `outputs/` 中缺少从 Google Drive 下载的前序 zip, comparison 会把 evidence path 缺失显式记录为 provenance 问题。
3. 该检查不改变小样本 evidence boundary, 也不把 GPU smoke 或小样本记录提升为正式 external baseline 论文结论。
4. 若需要关闭正式 baseline 结果缺口, 应先确保受治理结果包或官方复现 evidence paths 在当前审计边界内可解析, 再重建 comparison、evidence audit 和 submission readiness。
5. formal evidence path resolution 已支持显式外部镜像根目录, 例如通过 `--evidence-search-root` 或 `SLM_WM_EVIDENCE_SEARCH_ROOTS` 指向 Google Drive 的 `SLM` 目录; 仓库不会硬编码用户机器路径, 也不会因此改变 full-main、fixed-FPR、攻击矩阵检测和正式 claim 的接受边界。
6. 当前通过显式镜像根目录重建后, `formal_evidence_path_reference_count=28`, `search_resolved_formal_evidence_path_count=28`, `missing_formal_evidence_path_count=0`, `formal_evidence_path_resolution_ready=true`; 但 `formal_import_validation_ready=false`, 主要阻断仍是 `full_main_resource_profile_required`、`full_main_prompt_protocol_ready_required`、`fixed_fpr_baseline_calibration_ready_required` 与 `attack_matrix_baseline_detection_ready_required`。

### evidence closure entry review

1. 已新增 `evidence_closure_entry_review` 审计入口, 用于在进入论文投稿级证据闭合前汇总 submission readiness、external baseline、dataset-level quality 和 small-sample boundary。
2. 该入口只生成 `entry_review_report.json`、`entry_review_checklist.csv` 与 manifest, 不生成论文主表、主图或 supported claim。
3. 当前设计把 `entry_review_ready` 与 `evidence_closure_allowed` 分开: 即使报告可供用户审计, 只要 formal baseline、full-main 样本规模、fixed-FPR 重校准或 dataset-level FID / KID 未闭合, 就保持 `evidence_closure_allowed=false`。
4. 用户应审计该报告后再决定是否允许项目进入论文投稿级证据闭合; 若仍保持小样本约束, 则只能继续在受限证据边界内推进, 不得声明论文级结论。
