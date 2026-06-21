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
| construction_unit_name | `stage_12_threshold_calibration_metrics` |
| phase_status | `local_calibration_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/geometric_rescue/geometry_rescue_audit.json`; `outputs/geometric_rescue/aligned_detection_records.jsonl` |
| expected_output_manifest | `outputs/threshold_calibration/manifest.local.json` |
| expected_outputs | `outputs/threshold_calibration/calibration_thresholds.json`; `outputs/threshold_calibration/fixed_fpr_operating_points.csv`; `outputs/threshold_calibration/standard_watermark_metrics.csv`; `outputs/threshold_calibration/quality_metrics_summary.csv`; `outputs/threshold_calibration/roc_curve_points.csv`; `outputs/threshold_calibration/det_curve_points.csv`; `outputs/threshold_calibration/score_distribution_table.csv`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/threshold_calibration/rescue_fpr_audit.csv`; `outputs/threshold_calibration/manifest.local.json` |
| blocking_items | 褰撳墠 fixed-FPR 妗嗘灦鍙敱 governed records 閲嶅缓, 浣?`aligned_content_score` 浠嶆潵鑷湰鍦颁唬鐞? 鏈€鏂扮湡瀹?aligned rescoring 鍖呭凡鍚戜笅娓镐紶鎾?PSNR銆丼SIM銆丮SE銆丮AE銆丩PIPS 涓?CLIP score, FID / KID 浠嶆槸 dataset-level 鏈绠楁寚鏍? 鍥犳 `full_method_claim_ready=false`銆?|
| fallback_path | 鑻?rescue 鍚?evidence-level FPR 瓒呰繃鐩爣 operating point, 鍙厑璁镐繚鐣?raw content claim 鎴栧皢瀹屾暣绯荤粺 fixed-FPR 涓诲紶鏍囪涓?unsupported; 涓嶅厑璁稿彧鎶ュ憡 raw content FPR銆?|
| invariants | 鍐呭闃堝€煎彧鐢?calibration clean negative 鍐荤粨; test split 涓嶅弬涓庤皟闃堝€? clean negative 涓?attacked negative 鍒嗗紑瀹¤; rescue window 涓?fail reason gate 淇濇寔鍐荤粨銆?|
| next_stage_entry | 鍙互杩涘叆鏀诲嚮鐭╅樀涓庡啀鎵╂暎鏀诲嚮璁板綍鏋勫缓; 鑻ヨ鏀拺璁烘枃绾?fixed-FPR 涓诲紶, 浠嶉渶鎶婄湡瀹?aligned latent 閲嶅垽鎵╁睍鍒板畬鏁?calibration / test 瑙勬ā, 骞惰ˉ榻?dataset-level FID / KID 涓庣湡瀹炲浘鍍忔敾鍑婚棴鐜€?|

### stage12 宸插畬鎴愬唴瀹?

1. 鏇存柊 `experiments/protocol/calibration.py`, 鏂板 `FixedFprCalibrationConfig`銆乣FixedFprThreshold`銆乫ixed-FPR 闃堝€煎喕缁撱€佹牎鍑嗗悗鍒ゅ畾銆丄UC銆丷OC / DET 涓?score distribution 璁＄畻鍑芥暟銆?
2. 鏇存柊 `experiments/protocol/__init__.py`, 瀵煎嚭 fixed-FPR 鏍″噯鏍稿績瀵硅薄鍜屽嚱鏁般€?
3. 鏂板 `scripts/write_threshold_calibration_outputs.py`, 浠?`outputs/geometric_rescue/` 璁板綍閲嶅缓 calibration thresholds銆乷perating point銆乻tandard metrics銆乹uality metrics銆丷OC / DET銆乻core distribution銆乼hreshold degeneracy 鍜?rescue FPR audit銆?
4. 鏂板 `tests/functional/test_threshold_calibration.py`, 瑕嗙洊闃堝€煎彧鏉ヨ嚜 calibration clean negative銆乧lean negative 涓?attacked negative 鍒嗗紑瀹¤銆乽nsupported 璐ㄩ噺鎸囨爣涓嶄吉瑁呬负璁烘枃璇佹嵁銆?
5. `docs/field_registry.md` 宸茬櫥璁?fixed-FPR銆乷perating point銆丄UC銆丷OC / DET銆丗PR audit銆乵etric status 鍜岄槇鍊奸€€鍖栫浉鍏冲瓧娈点€?

### stage12 褰撳墠浜х墿鎽樿

1. `outputs/threshold_calibration/calibration_thresholds.json` 鏄剧ず `target_fpr=0.05`, `calibration_negative_count=14`, `observed_fpr=0.0`, `threshold_degenerate=false`, `threshold_value=0.5174190728458973`銆?
2. `outputs/threshold_calibration/fixed_fpr_operating_points.csv` 鏄剧ず `true_positive_rate=0.84375`, `raw_content_clean_fpr=0.03125`, `evidence_clean_fpr=0.03125`, `evidence_attacked_fpr=0.15625`銆?
3. `outputs/threshold_calibration/rescue_fpr_audit.csv` 鏄剧ず attacked negative 鐨?evidence-level FPR 瓒呰繃 `target_fpr=0.05`, 鍥犳瀹屾暣绯荤粺 fixed-FPR 涓诲紶蹇呴』淇濇寔 unsupported銆?
4. `outputs/threshold_calibration/quality_metrics_summary.csv` 宸茬敱鏈€鏂扮湡瀹?aligned rescoring 鍖呭埛鏂? PSNR=`28.774532071397005`, SSIM=`0.9903153991736182`, MSE=`0.0013260099804028869`, MAE=`0.02013162337243557`, LPIPS=`0.03199240565299988`, `clip_score=0.3809072971343994`; FID / KID 浠嶄繚鐣?`dataset_level_metric_not_computed_in_pair_run`銆?
5. `outputs/threshold_calibration/threshold_degeneracy_report.json` 涓?`raw_content_claim_ready=true`, 浣?`full_method_claim_ready=false`, `unsupported_reason=aligned_content_score_local_proxy`銆?

### stage12 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_threshold_calibration_outputs.py --aligned-rescoring-package-path outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip` | pass, `aligned_rescoring_quality_metrics_ready=true`, `real_aligned_rescore_count=3`, `evidence_attacked_fpr=0.15625` |
| `pytest tests/functional/test_threshold_calibration.py -q` | pass, 3 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_13_attack_matrix_regeneration

| item | value |
| --- | --- |
| construction_unit_name | `stage_13_attack_matrix_regeneration` |
| phase_status | `local_attack_matrix_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-20` |
| input_manifest | `outputs/geometric_rescue/manifest.local.json`; `outputs/threshold_calibration/manifest.local.json` |
| expected_output_manifest | `outputs/attack_matrix/manifest.local.json` |
| expected_outputs | `outputs/attack_matrix/attacked_images/`; `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/attacked_image_registry.jsonl`; `outputs/attack_matrix/attack_detection_records.jsonl`; `outputs/attack_matrix/attack_family_metrics.csv`; `outputs/attack_matrix/attack_strength_curve.csv`; `outputs/attack_matrix/score_retention_by_attack.csv`; `outputs/attack_matrix/rescue_by_attack.csv`; `outputs/attack_matrix/manifest.local.json` |
| blocking_items | 褰撳墠甯歌鏀诲嚮涓?record-level proxy, 鏈敓鎴愮湡瀹?attacked image 鏂囦欢; 鍐嶆墿鏁ｆ敾鍑婚渶瑕佺湡瀹?GPU 鍥惧儚閲嶇敓鎴愪骇鐗? 褰撳墠缁熶竴鏍囪涓?`unsupported`銆?|
| fallback_path | 甯歌鏀诲嚮鍙綔涓烘湰鍦板彲閲嶅缓鍗忚涓庤〃鏍奸摼璺? 鍐嶆墿鏁ｆ敾鍑讳繚鐣欓厤缃€乨igest 鍜?unsupported reason, 涓嶈繘鍏ヨ鏂囦富寮犮€?|
| invariants | 鏀诲嚮鍚庢娴嬪鐢?`stage_12` 鍐荤粨鐨?fixed-FPR 闃堝€笺€乺escue window 鍜?fail reason gate; clean negative 涓?attacked negative 鍒嗗紑缁熻; `full_method_claim_ready=false`; `supports_paper_claim=false`銆?|
| next_stage_entry | 鍙繘鍏ュ閮?baseline 瀵规瘮涓庡唴閮ㄦ秷铻嶈瘉鎹瀯寤? 鑻ヨ褰㈡垚 robustness 涓诲紶, 闇€瑕佺敤鐪熷疄 attacked image 鏂囦欢鍜岀湡瀹炲啀鎵╂暎鏀诲嚮浜х墿鏇挎崲鏈湴浠ｇ悊璁板綍銆?|

### stage13 宸插畬鎴愬唴瀹?

1. 鏂板 `experiments/protocol/attacks.py`, 瀹氫箟 `AttackConfig`銆乣AttackEvaluationBoundary`銆乣AttackDetectionRecord`銆侀粯璁ゆ敾鍑荤煩闃甸厤缃€佹敾鍑婚厤缃憳瑕併€乺ecord-level 鏀诲嚮浠ｇ悊銆乤ttack family metrics銆乻trength curve銆乻core retention 鍜?rescue-by-attack 鑱氬悎鍑芥暟銆?
2. 鏇存柊 `experiments/protocol/__init__.py`, 瀵煎嚭鏀诲嚮鐭╅樀鍗忚瀵硅薄鍜岃仛鍚堝嚱鏁般€?
3. 鏂板 `scripts/write_attack_matrix_outputs.py`, 浠?`outputs/geometric_rescue/aligned_detection_records.jsonl`銆乣outputs/geometric_rescue/manifest.local.json`銆乣outputs/threshold_calibration/calibration_thresholds.json`銆乣outputs/threshold_calibration/threshold_degeneracy_report.json` 鍜?`outputs/threshold_calibration/manifest.local.json` 閲嶅缓鏀诲嚮鐭╅樀浜х墿銆?
4. 褰撳墠榛樿鏀诲嚮鐭╅樀瑕嗙洊 JPEG compression銆丟aussian noise銆丟aussian blur銆乺esize銆乧rop銆乺otation銆乧rop-resize銆乧omposite geometric attacks, 鍚屾椂鐧昏 img2img regeneration銆丏DIM inversion + regeneration銆丼DEdit regeneration 鍜?diffusion purification銆?
5. 甯歌鏀诲嚮閰嶇疆鍐欏叆 `probe`銆乣pilot` 鍜?`full_main` 璧勬簮妗ｄ綅; 鍐嶆墿鏁ｆ敾鍑诲啓鍏?`full_extra` 璧勬簮妗ｄ綅骞朵繚鐣?`real_gpu_attack_required` unsupported reason銆?
6. 鏂板 `tests/functional/test_attack_matrix.py`, 瑕嗙洊鏀诲嚮閰嶇疆鎽樿绋冲畾鎬с€佸父瑙勬敾鍑诲垎鏁颁繚鎸佺巼涓嬮檷銆佸啀鎵╂暎鏀诲嚮 unsupported 杈圭晫銆佽剼鏈骇鐗╁彲閲嶅缓鍜?outputs 鐩綍绾︽潫銆?
7. `docs/field_registry.md` 宸茬櫥璁版敾鍑婚厤缃€佹敾鍑昏褰曘€乻ource / attacked digest銆乻core retention銆乹uality proxy銆乤ttention consistency銆佹敾鍑荤粺璁″拰 manifest 鐩稿叧瀛楁銆?

### stage13 褰撳墠浜х墿鎽樿

1. `outputs/attack_matrix/attack_manifest.json` 鏄剧ず `attack_config_count=14`, `attack_family_count=3`, `attack_record_count=1344`, `performed_attack_record_count=960`, `gpu_attack_unsupported_count=384`, `attack_metrics_ready=true`銆?
2. `outputs/attack_matrix/attack_family_metrics.csv` 宸插寘鍚父瑙勬敾鍑荤殑 `true_positive_rate`銆乣false_positive_rate`銆乣clean_false_positive_rate`銆乣attacked_false_positive_rate`銆乣quality_score_proxy_mean`銆乣score_retention_mean`銆乣lf_score_retention_mean`銆乣hf_score_retention_mean`銆乣attention_consistency_proxy_mean`銆乣geometry_reliable_rate` 鍜?`rescue_rate`銆?
3. 鍐嶆墿鏁ｆ敾鍑昏淇濈暀閰嶇疆涓?digest, 浣?`metric_status=unsupported`, `supported_record_count=0`, 涓嶆敮鎸佽鏂?robustness 涓诲紶銆?
4. `outputs/attack_matrix/attacked_images/` 褰撳墠涓虹┖鐩綍, 琛ㄧず鏈湴鏈敓鎴愮湡瀹?attacked image 鏂囦欢; `attacked_image_registry.jsonl` 鍙櫥璁板彈娌荤悊浠ｇ悊鎽樿銆?
5. `outputs/attack_matrix/attack_manifest.json` 涓?`outputs/attack_matrix/manifest.local.json` 宸茬户鎵挎渶鏂扮湡瀹?aligned rescoring 鍖呰矾寰勩€丼HA256 鎽樿銆乣aligned_rescoring_quality_metrics_ready=true`銆乣perceptual_metrics_ready=true` 涓?`real_aligned_rescore_count=3`銆?

### stage13 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄敾鍑荤煩闃靛崗璁€佽〃鏍奸噸寤洪摼璺拰甯歌鏀诲嚮鏈湴浠ｇ悊缁熻, 涓嶆槸姝ｅ紡 robustness 瀹為獙缁撹銆?
2. record-level proxy 鍙兘鐢ㄤ簬楠岃瘉瀛楁銆佺粺璁¤竟鐣屻€佽〃鏍煎舰鎬佸拰 artifact rebuild, 涓嶈兘鏇夸唬鐪熷疄鍥惧儚鏀诲嚮銆?
3. 鍐嶆墿鏁ｆ敾鍑诲繀椤诲湪鐪熷疄 GPU 鐜涓敓鎴?attacked image銆乻ource image digest銆乤ttack config digest 鍜屾娴嬭褰曞悗, 鎵嶈兘浠?`unsupported` 杩涘叆鍙粺璁＄姸鎬併€?
4. fixed-FPR 杈圭晫浠嶆部鐢?`stage_12` 鐨勭粨璁? raw content claim 鍙互灞€閮?ready, 瀹屾暣鏂规硶 `full_method_claim_ready=false`銆?

### stage13 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_attack_matrix_outputs.py` | pass, `attack_record_count=1344`, `performed_attack_record_count=960`, `gpu_attack_unsupported_count=384`, `attack_metrics_ready=true` |
| `pytest tests/functional/test_attack_matrix.py -q` | pass, 4 passed |
| `pytest -q` | pass, 86 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_14_external_baseline_comparison

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_comparison` |
| phase_status | `baseline_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json` |
| expected_output_manifest | `outputs/external_baseline_comparison/manifest.local.json` |
| expected_outputs | `outputs/external_baseline_comparison/baseline_observations.jsonl`; `outputs/external_baseline_comparison/baseline_metrics.csv`; `outputs/external_baseline_comparison/baseline_comparison_table.csv`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| blocking_items | 褰撳墠鍙畬鎴愬閮?baseline 鐨勫崗璁?adapter 涓庡叕骞冲姣旇〃鏍奸摼璺? 灏氭湭鎺ュ叆瀹樻柟浠ｇ爜銆佸鐜板疄楠岀粨鏋滄垨鍙楁不鐞嗗鍏ョ粨鏋? 鍥犳鎵€鏈夊閮?baseline 鎸囨爣淇濇寔 `unsupported`銆?|
| fallback_path | 澶栭儴 baseline 鏃犵粨鏋滄椂鍙櫥璁?`external_baseline_result_missing`, 涓嶆墜宸ュ～鍐欐寚鏍? 褰撳墠鏂规硶琛屼篃鍙繚鐣?attack matrix local proxy, 涓嶄綔涓鸿鏂囩骇 superiority 鎴?robustness 涓诲紶銆?|
| invariants | baseline 涓?SLM-WM 蹇呴』鍏变韩 prompt 鍗忚銆佹敾鍑荤煩闃靛崗璁拰 fixed-FPR operating point; unsupported baseline 涓嶈繘鍏ヤ富缁撹; 鎵€鏈夋柊浜х墿淇濇寔 `supports_paper_claim=false`銆?|
| next_stage_entry | 鍙繘鍏ュ唴閮ㄦ秷铻嶈瘉鎹瀯寤? 鑻ヨ褰㈡垚璁烘枃绾у閮ㄥ姣? 闇€鍚庣画鎺ュ叆 baseline 瀹樻柟浠ｇ爜鎴栧彈娌荤悊瀵煎叆缁撴灉, 骞跺湪鐩稿悓鍗忚涓嬮噸寤鸿〃鏍笺€?|

### stage14 宸插畬鎴愬唴瀹?

1. 鏂板 `experiments/baselines/adapters.py`, 瀹氫箟 `BaselineSpec`銆乣BaselineObservation`銆侀粯璁ゅ閮?baseline 娓呭崟銆乥aseline 瑙傛祴璁板綍鏋勯€犮€乥aseline 鎸囨爣鑱氬悎鍜屽悓鍗忚瀵规瘮琛ㄦ瀯閫犮€?
2. 鏂板 `scripts/write_external_baseline_comparison_outputs.py`, 浠?`outputs/attack_matrix/` 涓?`outputs/threshold_calibration/` 璇诲彇鍙楁不鐞嗚緭鍏? 閲嶅缓澶栭儴 baseline observations銆乵etrics銆乧omparison table銆乺untime report 鍜?manifest銆?
3. 榛樿鐧昏 8 涓閮?baseline: Tree-Ring銆丟aussian Shading銆丼hallow Diffuse銆乀2SMark銆丼table Signature銆丷ivaGAN銆乀rustMark 鍜?Watermark Anything銆?
4. 鏂板鏍圭洰褰?`external_baseline/`, 鐢ㄤ簬鏈湴淇濆瓨澶栭儴 baseline 瀹樻柟婧愮爜鎴栧鐜伴暅鍍? 涓昏〃婧愮爜妲戒綅鍖呮嫭 Tree-Ring銆丟aussian Shading銆丼hallow Diffuse銆乀2SMark, 琛ュ厖琛ㄦ簮鐮佹Ы浣嶅寘鎷?Stable Signature銆丷ivaGAN銆乀rustMark銆乄atermark Anything銆?
5. 褰撳墠 baseline adapter 鍙喕缁撳叕骞冲崗璁竟鐣? 涓嶈繍琛屾垨浼€犲閮ㄦ柟娉曠粨鏋? 鎵€鏈夊閮?baseline 琛屽潎鍐欏叆 `metric_status=unsupported` 涓?`unsupported_reason=external_baseline_result_missing`銆?
6. 鏂板 `tests/functional/test_external_baseline_comparison.py`, 瑕嗙洊榛樿 baseline 娓呭崟銆佷骇鐗╅噸寤恒€乧laim 瀹夊叏杈圭晫鍜?outputs 鐩綍绾︽潫銆?
7. `docs/field_registry.md` 宸茬櫥璁?baseline observation銆乥aseline readiness銆佸叡鍚屽崗璁€乧omparison table銆乺untime report 鍜屽閮ㄦ簮鐮佹潵婧愮櫥璁扮浉鍏冲瓧娈点€?

### stage14 褰撳墠浜х墿鎽樿

1. `outputs/external_baseline_comparison/baseline_runtime_report.json` 鏄剧ず `baseline_count=8`, `baseline_observation_count=112`, `comparable_baseline_count=8`, `baseline_result_ready_count=0`, `comparison_protocol_ready=true`, `baseline_results_ready=false`銆?
2. `outputs/external_baseline_comparison/baseline_metrics.csv` 涓?8 涓?baseline 鍧囦负 `baseline_adapter_ready=True`, 浣?`baseline_official_code_ready=False`, `baseline_reproduced_result_ready=False`, `baseline_imported_result_ready=False`銆?
3. `outputs/external_baseline_comparison/baseline_comparison_table.csv` 鍖呭惈 `slm_wm_current` 鏈湴浠ｇ悊琛屽拰 8 涓閮?baseline unsupported 琛? 鎵€鏈夎鍧囦负 `supports_paper_claim=False`銆?
4. `outputs/external_baseline_comparison/manifest.local.json` 璁板綍杈撳叆銆佽緭鍑恒€乥aseline spec 鎽樿銆乻ummary 鎽樿銆佷唬鐮佺増鏈拰閲嶅缓鍛戒护 `python scripts/write_external_baseline_comparison_outputs.py`銆?

### stage14 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄閮?baseline 鍏钩瀵规瘮鍗忚銆乻chema adapter銆佸彈娌荤悊琛ㄦ牸閾捐矾鍜岀己澶辩粨鏋滆竟鐣? 涓嶆槸澶栭儴 baseline 瀹炴祴鎬ц兘缁撹銆?
2. 褰撳墠澶栭儴 baseline 鎸囨爣涓嶅緱鐢ㄤ簬璁烘枃涓昏〃缁撹, 鍙兘璇存槑瀵规瘮鍗忚宸茬粡鍙璁°€佸彲澶嶇幇骞剁瓑寰呯湡瀹?baseline 缁撴灉鎺ュ叆銆?
3. 鍚庣画鑻ユ帴鍏ュ畼鏂逛唬鐮佹垨瀵煎叆缁撴灉, 蹇呴』鐢?adapter 鎴栧彈娌荤悊瀵煎叆鏂囦欢鐢熸垚 records, 涓嶅緱鎵嬪伐琛ヨ〃銆?

### stage14 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_external_baseline_comparison_outputs.py` | pass, `baseline_count=8`, `baseline_observation_count=112`, `baseline_result_ready_count=0` |
| `pytest tests/functional/test_external_baseline_comparison.py -q` | pass, 2 passed |
| `pytest -q` | pass, 88 passed |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

## stage_15_internal_ablation_evidence

| item | value |
| --- | --- |
| construction_unit_name | `internal_ablation_evidence` |
| phase_status | `ablation_protocol_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| expected_output_manifest | `outputs/internal_ablation_evidence/manifest.local.json` |
| expected_outputs | `outputs/internal_ablation_evidence/ablation_records.jsonl`; `outputs/internal_ablation_evidence/mechanism_ablation_table.csv`; `outputs/internal_ablation_evidence/method_pairwise_delta_table.csv`; `outputs/internal_ablation_evidence/ablation_by_attack_family.csv`; `outputs/internal_ablation_evidence/ablation_claim_summary.json`; `outputs/internal_ablation_evidence/manifest.local.json` |
| blocking_items | 褰撳墠鍐呴儴娑堣瀺澶嶇敤 attack matrix 鐨?record-level proxy, 鐪熷疄 attacked image 涓庡啀鎵╂暎鏀诲嚮 GPU 闂幆浠嶆湭琛ラ綈; 澶栭儴 baseline 鐪熷疄缁撴灉浠嶄负 `baseline_results_ready=false`, 鍥犳澶栭儴 superiority 涓庡畬鏁?robustness 涓诲紶浠嶄笉鑳芥垚绔嬨€?|
| fallback_path | 鍐呴儴娑堣瀺鍙敤浜庡喕缁撴満鍒跺繀瑕佹€у崗璁€侀€€鍖栭摼鍜岃〃鏍奸噸寤洪摼璺? 涓嶆妸鏈湴浠ｇ悊娑堣瀺缁撴灉鍐欐垚璁烘枃 supported claim銆?|
| invariants | 姣忎釜娑堣瀺蹇呴』鐪熷疄鏀瑰彉鏈哄埗瀛楁鎴栧垽瀹氳竟鐣? `full_slm_wm` 涓哄弬鑰冭; `geo_direct_positive_audit` 鍙兘浣滀负瀹¤鍙嶄緥; 鎵€鏈変骇鐗╀繚鎸?`supports_paper_claim=false`銆?|
| next_stage_entry | 鍙繘鍏ヨ鏂囦骇鐗╄瘉鎹璁? 浣嗗璁＄粨璁哄繀椤讳繚鐣?local proxy 杈圭晫, 骞舵妸鐪熷疄鍥惧儚鏀诲嚮銆佸閮?baseline 瀹炴祴涓?full-main 瑙勬ā缁熻鍒椾负鍚庣画琛ヨ瘉浠诲姟銆?|

### stage15 宸插畬鎴愬唴瀹?

1. 鏂板 `experiments/ablations/mechanisms.py`, 瀹氫箟 `AblationSpec`銆侀粯璁ゅ唴閮ㄦ秷铻嶆竻鍗曘€佹秷铻?records 鏋勯€犮€佹満鍒惰〃鑱氬悎銆佹寜鏀诲嚮鏃忚仛鍚堛€乸airwise delta 鍜?claim summary 鏋勯€犮€?
2. 鏂板 `scripts/write_internal_ablation_outputs.py`, 浠庢敾鍑荤煩闃点€侀槇鍊兼牎鍑嗗拰澶栭儴 baseline 瀵规瘮 manifest 璇诲彇鍙楁不鐞嗚緭鍏? 閲嶅缓鍐呴儴娑堣瀺 records銆佹満鍒惰〃銆乸airwise delta銆乤ttack-family 琛ㄣ€乧laim summary 鍜?manifest銆?
3. 榛樿鐧昏 17 涓唴閮ㄦ秷铻? Full SLM-WM銆丟lobal Null Space銆丯o Semantic Mask銆丯o Semantic JVP銆丯o Risk Weight銆丷andom Basis銆丩F-only銆丠F-only銆丯o-HF銆丯o-LF銆丯o Tail Truncation銆丗FT-sync-only銆両mage-registration-only銆丯o Attention Anchor銆丯o Rescue銆丯o Attestation 鍜?Geo-direct-positive audit銆?
4. `full_slm_wm` 淇濇寔涓婃父鏀诲嚮璁板綍鐨勫畬鏁存柟娉曞垽瀹? 鍏朵粬娑堣瀺閫氳繃 LF/HF retention銆乤ligned gain銆乤ttention consistency銆乬eometry reliability銆乺escue gate銆乤ttestation gate 鎴?content gate 鍙嶄緥璺緞浜х敓瀹為檯瀛楁鍙樺寲銆?
5. 鏂板 `tests/functional/test_internal_ablation_evidence.py`, 瑕嗙洊娑堣瀺娓呭崟瀹屾暣鎬с€佸叧閿満鍒跺疄闄呭彉鍖栥€佽緭鍑虹洰褰曠害鏉熴€乧laim 瀹夊叏杈圭晫鍜岃〃鏍煎彲閲嶅缓鎬с€?
6. `docs/field_registry.md` 宸茬櫥璁板唴閮ㄦ秷铻?records銆佹満鍒惰〃銆乸airwise delta銆乧laim summary 鍜?manifest 鐩稿叧瀛楁銆?

### stage15 褰撳墠浜х墿鎽樿

1. `outputs/internal_ablation_evidence/ablation_claim_summary.json` 鏄剧ず `ablation_count=17`, `ablation_record_count=22848`, `mechanism_group_count=7`, `ablation_protocol_ready=true`, `mechanism_coverage_ready=true`, `attack_metrics_ready=true`, `external_baseline_result_ready=false`銆?
2. `outputs/internal_ablation_evidence/mechanism_ablation_table.csv` 鍖呭惈 17 涓秷铻嶈, 姣忚鍧囪褰?TPR銆丗PR銆乻core retention銆乹uality proxy銆乤ttention consistency銆乬eometry reliability銆乺escue rate銆乤ttestation availability 鍜岀浉瀵瑰畬鏁存柟娉曠殑 delta銆?
3. `outputs/internal_ablation_evidence/method_pairwise_delta_table.csv` 鍖呭惈 96 鏉＄浉瀵?`full_slm_wm` 鐨勬寚鏍囧樊寮傝褰曘€?
4. `outputs/internal_ablation_evidence/ablation_by_attack_family.csv` 鍖呭惈 51 鏉℃寜娑堣瀺鍜屾敾鍑绘棌鑱氬悎鐨勯€€鍖栬褰曘€?
5. 鎵€鏈夊唴閮ㄦ秷铻嶄骇鐗╁潎淇濇寔 `supports_paper_claim=false`, `full_method_claim_ready=false`銆?

### stage15 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄唴閮ㄦ秷铻嶅崗璁€佹満鍒堕€€鍖栭摼銆佽〃鏍奸噸寤洪摼璺拰 claim 瀹夊叏杈圭晫, 涓嶆槸璁烘枃绾ф渶缁堟秷铻嶇粨璁恒€?
2. 褰撳墠娑堣瀺缁撴灉澶嶇敤 record-level attack proxy, 涓嶈兘鏇夸唬鐪熷疄鍥惧儚鏀诲嚮鍜?full-main 瑙勬ā缁熻銆?
3. `geo_direct_positive_audit` 鏄庣‘鏄?content gate 鍙嶄緥瀹¤, 涓嶅緱浣滀负姝ｅ紡鏂规硶鎴栦富琛ㄦ柟娉曡銆?
4. `no_attestation` 浼氳妫€娴嬭瘉鎹笉鑳借繘鍏ュ彲瀹¤鏂规硶涓诲紶, 鐢ㄤ簬璇佹槑 attestation gate 鐨勫繀瑕佹€с€?

### stage15 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_internal_ablation_outputs.py` | pass, `ablation_count=17`, `ablation_record_count=22848`, `mechanism_coverage_ready=true` |
| `pytest tests/functional/test_internal_ablation_evidence.py -q` | pass, 3 passed |
| `pytest -q` | pass, 91 passed, 2 deselected |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_16_paper_artifact_evidence_audit

| item | value |
| --- | --- |
| construction_unit_name | `paper_artifact_evidence_audit` |
| phase_status | `evidence_gap_report_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/threshold_calibration/threshold_degeneracy_report.json`; `outputs/threshold_calibration/manifest.local.json`; `outputs/attack_matrix/attack_manifest.json`; `outputs/attack_matrix/manifest.local.json`; `outputs/external_baseline_comparison/manifest.local.json`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/internal_ablation_evidence/manifest.local.json`; `outputs/internal_ablation_evidence/ablation_claim_summary.json` |
| expected_output_manifest | `outputs/paper_artifact_evidence_audit/manifest.local.json` |
| expected_outputs | `outputs/paper_artifact_evidence_audit/claim_audit_table.csv`; `outputs/paper_artifact_evidence_audit/paper_table_readiness.csv`; `outputs/paper_artifact_evidence_audit/paper_figure_readiness.csv`; `outputs/paper_artifact_evidence_audit/evidence_gap_list.csv`; `outputs/paper_artifact_evidence_audit/artifact_builder_readiness_report.json`; `outputs/paper_artifact_evidence_audit/evidence_audit_dry_run.json`; `outputs/paper_artifact_evidence_audit/submission_blocker_report.json`; `outputs/paper_artifact_evidence_audit/manifest.local.json` |
| blocking_items | `submission_ready=false`; critical gaps 鍖呮嫭鐪熷疄 attacked image 闂幆銆佸啀鎵╂暎绫绘敾鍑荤湡瀹?GPU 楠岃瘉銆佸閮?baseline 缁撴灉銆乫ull-main 鏍锋湰瑙勬ā; major gaps 鍖呮嫭瀹屾暣鏂规硶 fixed-FPR 閲嶆牎鍑嗗拰 dataset-level FID / KID銆?|
| fallback_path | 褰撳墠浠呭喕缁?artifact builder 涓?evidence audit 閾捐矾, 涓嶅喕缁撴姇绋跨粨鏋? 涓嶆妸棰勮琛ㄦ牸鎴栨湰鍦颁唬鐞嗙粨鏋滃啓鎴愯鏂囩骇缁撹銆?|
| invariants | 鎵€鏈夋柊浜х墿淇濇寔 `supports_paper_claim=false`; 涓嶆墜宸ヨˉ琛? Notebook 涓嶈兘鐩存帴鍐欐寮?records銆乼ables銆乫igures 鎴?reports; `main/` 涓嶇粦瀹氬灞傝繍琛岀洰褰曘€?|
| next_stage_entry | 闇€瑕佸厛鎸?`outputs/paper_artifact_evidence_audit/evidence_gap_list.csv` 琛ラ綈鐪熷疄鏀诲嚮闂幆銆佸閮?baseline 缁撴灉銆乫ull-main 缁熻鍜岃川閲忔暟鎹泦鎸囨爣, 鍐嶈繘鍏ユ姇绋垮喕缁撱€?|

### stage16 宸插畬鎴愬唴瀹?

1. 鏂板 `main/analysis/paper_evidence_audit.py`, 灏嗕笂娓?threshold銆乤ttack matrix銆乪xternal baseline 涓?internal ablation 浜х墿姹囨€讳负 claim audit銆佽〃鏍?readiness銆佸浘鏁版嵁 readiness銆佽瘉鎹己鍙ｅ拰鎶曠闃绘柇鎽樿銆?
2. 鏂板 `scripts/write_paper_artifact_evidence_audit_outputs.py`, 浠庡彈娌荤悊涓婃父 manifest 鍜?report 閲嶅缓 8 涓湰鍦板璁′骇鐗? 骞跺啓鍏?`outputs/paper_artifact_evidence_audit/`銆?
3. 鏂板 `tests/functional/test_paper_artifact_evidence_audit.py`, 楠岃瘉 claim 杈圭晫銆佺己鍙ｅ垪琛ㄣ€佽緭鍑虹洰褰曠害鏉熴€乵anifest 閲嶅缓鍜?`supports_paper_claim=false` 瀹夊叏杈圭晫銆?
4. 鏇存柊 `docs/field_registry.md`, 鐧昏 claim audit銆乸aper readiness銆乬ap list銆乥uilder readiness 涓?blocker report 鐩稿叧瀛楁銆?

### stage16 褰撳墠浜х墿鎽樿

1. `artifact_builder_readiness_report.json` 鏄剧ず `artifact_builder_ready=true`, `paper_artifact_audit_ready=true`, `claim_audit_row_count=7`, `table_readiness_row_count=6`, `figure_readiness_row_count=5`, `rebuildable_artifact_count=10`, `blocked_artifact_count=1`, `paper_ready_artifact_count=0`銆?
2. `submission_blocker_report.json` 鏄剧ず `submission_ready=false`, `blocking_claim_count=5`, `critical_gap_count=4`, `gap_count=6`銆?
3. `evidence_gap_list.csv` 灏嗙湡瀹?attacked image 闂幆銆佸啀鎵╂暎绫绘敾鍑荤湡瀹?GPU 楠岃瘉銆佸閮?baseline 缁撴灉銆乫ull-main 鏍锋湰瑙勬ā銆佸畬鏁存柟娉?fixed-FPR 閲嶆牎鍑嗗拰 dataset-level FID / KID 鍒椾负鍚庣画琛ヨ瘉椤广€?
4. `claim_audit_table.csv` 鏄庣‘澶栭儴 baseline superiority 涓?submission-ready package 浠嶄负 `unsupported`, 鏀诲嚮椴佹鎬т笌鍐呴儴鏈哄埗蹇呰鎬т粛涓?`preview_only`銆?

### stage16 瀹屾垚杈圭晫

1. 鏈樁娈靛畬鎴愮殑鏄鏂囦骇鐗╄瘉鎹璁￠摼璺拰 artifact builder readiness, 涓嶆槸璁烘枃绾?robustness銆乥aseline superiority 鎴?submission-ready 缁撹銆?
2. 褰撳墠 `paper_ready_artifact_count=0`, 鍥犱负 full-main 缁熻銆佺湡瀹炲浘鍍忔敾鍑婚棴鐜€佸閮?baseline 瀹炴祴缁撴灉鍜?dataset-level FID / KID 灏氭湭琛ラ綈銆?
3. 褰撳墠浜х墿鍙綔涓哄悗缁ˉ璇佷换鍔℃竻鍗曞拰鑷姩閲嶅缓鍏ュ彛, 涓嶈兘浣滀负璁烘枃涓昏〃銆佷富鍥炬垨鏈€缁?claim 鐨勭洿鎺ヨ瘉鎹€?

### stage16 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python -m py_compile main/analysis/paper_evidence_audit.py scripts/write_paper_artifact_evidence_audit_outputs.py tests/functional/test_paper_artifact_evidence_audit.py` | pass |
| `python tools/harness/inspect_repository.py .` | pass |
| `python scripts/write_paper_artifact_evidence_audit_outputs.py` | pass, `claim_audit_row_count=7`, `table_readiness_row_count=6`, `figure_readiness_row_count=5`, `gap_count=6`, `submission_ready=false` |
| `pytest tests/functional/test_paper_artifact_evidence_audit.py -q` | pass, 2 passed |
| `pytest -q` | pass, 93 passed, 2 deselected |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |


## stage_17_pilot_full_submission_freeze

| item | value |
| --- | --- |
| construction_unit_name | `submission_readiness_gate` |
| phase_status | `blocked_by_evidence_gaps` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/paper_artifact_evidence_audit/manifest.local.json`; `outputs/paper_artifact_evidence_audit/artifact_builder_readiness_report.json`; `outputs/paper_artifact_evidence_audit/submission_blocker_report.json`; `outputs/paper_artifact_evidence_audit/evidence_gap_list.csv`; `docs/extraction_profiles.md`; `docs/release_boundary.md` |
| expected_output_manifest | `outputs/submission_readiness/submission_readiness_manifest.local.json` |
| expected_outputs | `outputs/submission_readiness/readiness_blocker_report.json`; `outputs/submission_readiness/required_evidence_inputs.csv`; `outputs/submission_readiness/release_profile_dry_run.csv`; `outputs/submission_readiness/submission_readiness_manifest.local.json` |
| blocking_items | `readiness_decision=blocked`; `submission_ready=false`; `required_input_count=6`; `critical_required_input_count=4`; `paper_ready_artifact_count=0`銆?|
| fallback_path | 鍙敓鎴愭姇绋垮氨缁樆鏂姤鍛婂拰 release dry-run 娓呭崟, 涓嶅鍑烘姇绋垮€欓€夊寘, 涓嶅喕缁撹鏂囩骇琛ㄦ牸銆佸浘鎴?report銆?|
| invariants | stage16 evidence audit 鏈€氳繃鎶曠鍐荤粨鍓? release dry-run 鍙繍琛屼笉绛変环浜庢姇绋垮氨缁? 鎵€鏈夋柊澧炰骇鐗╀繚鎸?`supports_paper_claim=false`; 涓嶆墜宸ヨˉ琛ㄦ垨鎵嬪伐鏍囪 claim銆?|
| next_stage_entry | 闇€瑕佸厛琛ラ綈鐪熷疄 attacked image 闂幆銆佸啀鎵╂暎绫绘敾鍑荤湡瀹?GPU 楠岃瘉銆佸閮?baseline 缁撴灉銆乫ull-main 鏍锋湰瑙勬ā銆佸畬鏁存柟娉?fixed-FPR 閲嶆牎鍑嗗拰 dataset-level FID / KID, 鍐嶉噸鏂拌繍琛屾湰闂ㄧ銆?|

### stage17 褰撳墠鎺ㄨ繘鍐呭

1. 鏂板 `main/analysis/submission_readiness.py`, 灏?stage16 璇佹嵁瀹¤浜х墿銆佽瘉鎹己鍙ｅ拰 release dry-run 鎽樿鍚堟垚涓烘姇绋垮氨缁棬绂佸垽瀹氥€?
2. 鏂板 `scripts/write_submission_readiness_outputs.py`, 浠?`outputs/paper_artifact_evidence_audit/` 璇诲彇鍙楁不鐞嗚緭鍏? 鐢熸垚闃绘柇鎶ュ憡銆佸緟琛ラ綈杈撳叆娓呭崟銆乺elease profile dry-run 琛ㄥ拰 manifest銆?
3. 鏂板 `tests/functional/test_submission_readiness.py`, 楠岃瘉瀛樺湪璇佹嵁缂哄彛鏃朵笉寰楀厑璁告姇绋垮喕缁? 骞堕獙璇佽緭鍑虹洰褰曘€乵anifest 涓?`supports_paper_claim=false` 杈圭晫銆?
4. 鏇存柊 `docs/field_registry.md`, 鐧昏鎶曠灏辩华闂ㄧ銆佸緟琛ラ綈杈撳叆鍜?release dry-run 鐩稿叧瀛楁銆?

### stage17 褰撳墠浜х墿鎽樿

1. `readiness_blocker_report.json` 鏄剧ず `readiness_decision=blocked`, `submission_ready=false`, `package_freeze_allowed=false`, `release_dry_run_ready=true`銆?
2. `required_evidence_inputs.csv` 鍖呭惈6涓緟琛ラ綈杈撳叆, 鍏朵腑4涓负 critical: 鐪熷疄 attacked image 闂幆銆佸啀鎵╂暎绫绘敾鍑荤湡瀹?GPU 楠岃瘉銆佸閮?baseline 缁撴灉鍜?full-main 鏍锋湰瑙勬ā銆?
3. `release_profile_dry_run.csv` 鏄剧ず `minimal_method_package` 涓?`paper_artifact_rebuild_package` 鐨?dry-run 鍙敓鎴愭枃浠舵竻鍗? 浣?`release_package_allowed=false`銆?
4. 褰撳墠涓嶈兘杩涘叆 submission-ready 鐘舵€? 鍥犱负璇佹嵁缂哄彛灏氭湭鍏抽棴涓?`paper_ready_artifact_count=0`銆?

### stage17 褰撳墠瀹屾垚杈圭晫

1. 鏈瀹屾垚鐨勬槸鎶曠灏辩华闂ㄧ鐨勯樆鏂璁￠摼璺? 涓嶆槸 stage17 瀹屾暣鎶曠鍐荤粨銆?
2. 鏈涓嶈繍琛?full-main 鎴?full-extra, 涓嶅鍑?release package, 涓嶇敓鎴愯鏂囩骇鏈€缁堣〃鍥俱€?
3. 鍚庣画鑻ヨˉ榻?evidence gap, 搴斿厛閲嶆柊杩愯 stage16 artifact evidence audit, 鍐嶉噸鏂拌繍琛屾湰闂ㄧ銆?

### stage17 褰撳墠楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python -m py_compile main/analysis/submission_readiness.py scripts/write_submission_readiness_outputs.py tests/functional/test_submission_readiness.py` | pass |
| `python scripts/write_submission_readiness_outputs.py` | pass, `readiness_decision=blocked`, `required_input_count=6`, `release_dry_run_ready=true` |
| `pytest tests/functional/test_submission_readiness.py -q` | pass, 2 passed |
| `pytest -q` | pass, 95 passed, 2 deselected |
| `python tools/harness/run_all_audits.py` | pass, 8/8 audits passed |

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
| phase_status | `colab_workflow_ready_gpu_run_required` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/aligned_rescoring/aligned_rescoring_manifest.local.json`; `outputs/attack_matrix/manifest.local.json`; `outputs/threshold_calibration/manifest.local.json` |
| expected_output_manifest | `outputs/real_attack_evaluation/real_attack_manifest.local.json` |
| expected_outputs | `paper_workflow/real_attack_evaluation_run.ipynb`; `paper_workflow/colab_utils/real_attack_evaluation.py`; `outputs/real_attack_evaluation/attacked_images/*.png`; `outputs/real_attack_evaluation/real_attack_detection_records.jsonl`; `outputs/real_attack_evaluation/real_attacked_image_registry.jsonl`; `outputs/real_attack_evaluation/real_attack_family_metrics.csv`; `outputs/real_attack_evaluation/real_attack_environment_report.json`; `outputs/real_attack_evaluation/real_attack_manifest.local.json`; `GoogleDrive/SLM/real_attack_evaluation/real_attack_evaluation_package_<utc>_<short_commit>.zip` |
| blocking_items | 鏈湴鐜鏃?GPU 鍜岀湡瀹?SD3.5 Medium 鏉冮噸, 鍥犳鏈鍙兘琛ラ綈 Colab 鐪熷疄 GPU workflow銆佸彈娌荤悊瀛楁銆佹墦鍖呭叆鍙ｅ拰杞婚噺娴嬭瘯; 鐪熷疄 attacked image 鏂囦欢涓?img2img / DDIM inversion / SDEdit / diffusion purification 瀹炴祴浠嶉渶瑕佸湪 Colab GPU 鎵ц notebook 鍚庡洖浼犲寘瀹¤銆?|
| fallback_path | 鑻ョ己灏?aligned rescoring 鍖呫€丠F_TOKEN銆丟PU runtime銆乮mage-to-image pipeline 鎴栨煇涓啀鎵╂暎鏀诲嚮鍚庣, helper 浼氬啓鍑?`run_decision=fail`銆乣unsupported_reason` 鍜岀幆澧冨揩鐓? 涓嶄細浼€犵湡瀹?attacked image 缁撴灉銆?|
| invariants | Notebook 鍙綔涓鸿繙绋嬪叆鍙? 姝ｅ紡閫昏緫浣嶄簬 `paper_workflow/colab_utils/real_attack_evaluation.py`; 鎵€鏈夋柊澧炰骇鐗╀繚鎸?`supports_paper_claim=false`, 鐩村埌閲嶆柊杩愯 attack matrix銆乫ixed-FPR 鏍″噯鍜岃鏂囪瘉鎹璁°€?|
| next_stage_entry | Colab 鐢熸垚骞跺洖浼?`real_attack_evaluation_package_<utc>_<short_commit>.zip` 鍚? 鏈湴搴斿厛瀹¤ records銆乺egistry銆乤ttacked images銆乵etrics銆乪nvironment report 鍜?manifest, 鍐嶅喅瀹氭槸鍚﹀叧闂湡瀹炲浘鍍忕骇鏀诲嚮闂幆涓庡啀鎵╂暎 GPU 楠岃瘉缂哄彛銆?|

### real attack evaluation workflow 宸插畬鎴愬唴瀹?

1. 鏂板 `paper_workflow/colab_utils/real_attack_evaluation.py`, 鏀寔浠?aligned rescoring 杈撳嚭鍥惧儚璇诲彇 source image, 鐢熸垚鐪熷疄 attacked image 鏂囦欢, 鐧昏 source / attacked image SHA256 digest, 骞跺啓鍑虹湡瀹炴敾鍑绘娴?records銆乤ttacked image registry銆乤ttack family metrics銆乪nvironment report 鍜?manifest銆?
2. 鏂板 `paper_workflow/real_attack_evaluation_run.ipynb`, 鏀寔 Colab 鍐峰惎鍔? 鎸傝浇 Google Drive銆佸畨瑁呭綋鍓?Colab 鍙繍琛屼緷璧栫粍鍚堛€佹媺鍙栦粨搴撱€佽鍙?`HF_TOKEN`銆佽В鍘嬪墠搴?aligned rescoring 鍖呫€佹鏌?GPU銆佸姞杞?SD3.5 Medium image-to-image pipeline銆佹墽琛?img2img銆丏DIM inversion銆丼DEdit 鍜?diffusion purification 鏀诲嚮, 骞舵妸缁撴灉鍖呬繚瀛樺埌 Drive銆?
3. 鏂板 `package_real_attack_evaluation_outputs`, 浼氭妸 attacked images銆乺ecords銆乺egistry銆乵etrics銆乪nvironment report銆乵anifest銆丯otebook銆乭elper 鍜屽叧閿笂娓?manifest 绾冲叆 zip銆?
4. 鏂板杞婚噺娴嬭瘯 `tests/functional/test_real_attack_evaluation.py`, 浣跨敤 mock pipeline 楠岃瘉 registry / digest / 妫€娴嬭褰?/ 鎵撳寘杈圭晫, 涓嶅湪榛樿 pytest 涓Е鍙戠湡瀹?GPU 鎺ㄧ悊銆?
5. 鏇存柊 `tests/constraints/test_notebook_entrypoint_contract.py`, 瑕嗙洊 Notebook 濮旀墭銆丏rive 璺緞銆佹棤鎵ц杈撳嚭銆佸姩鎬佷緷璧栧懡浠ゅ拰鎵撳寘浜х墿鏍稿銆?
6. 鏇存柊 `docs/field_registry.md`, 鐧昏鐪熷疄鏀诲嚮闂幆銆乤ttacked image 鏂囦欢銆乨igest 娉ㄥ唽銆佹敾鍑诲悗妫€娴嬪拰鍐嶆墿鏁?GPU 楠岃瘉鐘舵€佺浉鍏冲瓧娈点€?

### real attack evaluation workflow 褰撳墠杈圭晫

1. 鏈湴浠ｇ爜鍙樻洿涓嶈兘鐩存帴璇佹槑鐪熷疄 attacked image 缂哄彛宸插叧闂? 闇€瑕?Colab GPU 杩愯鍚庣殑 zip 鍖呬綔涓鸿瘉鎹€?
2. 褰撳墠鏀诲嚮鍚庢娴嬩娇鐢?`real_image_quality_proxy_after_attack` 鍙楁不鐞嗕唬鐞嗗垎鏁? 浣滅敤鏄畬鎴愮湡瀹炲浘鍍忔枃浠堕棴鐜拰閲嶆柊妫€娴嬭褰? 涓嶇瓑浠蜂簬璁烘枃绾?robustness 缁撹銆?
3. DDIM inversion 璺緞浼氬皾璇曚娇鐢?diffusers 鐨?`DDIMScheduler`; 鑻ュ綋鍓?SD3.5 image-to-image 鍚庣涓嶆帴鍙楄 scheduler, helper 浼氳褰?unsupported, 鑰屼笉鏄檷绾т吉閫?inversion 缁撴灉銆?
4. Colab 鍖呭洖浼犲悗, 浠嶉渶鎶婄湡瀹炴敾鍑荤粨鏋滃悜 attack matrix銆乼hreshold calibration銆乸aper artifact evidence audit 鍜?submission readiness gate 閲嶆柊浼犳挱銆?

### real attack evaluation workflow 杩藉姞淇

1. Notebook 宸叉敼涓轰粠 Google Drive 涓煡鎵惧苟閫夋嫨鎬цВ鍘嬪墠搴?`aligned_rescoring_package_*.zip` 涓?`threshold_calibration_package_*.zip`, 涓嶅啀渚濊禆鏈湴 `outputs/` 涓殑鍓嶅簭 zip, 涔熶笉鍐嶆妸鍖呭唴浠ｇ爜鏂囦欢瑕嗙洊鍥炰粨搴撳伐浣滃尯銆?
2. Notebook 宸叉敼涓哄厛杩愯 workflow 骞舵墦鍖呭埌 Drive, 鍐嶆墽琛屾柇瑷€; 鍗充娇鐪熷疄 GPU 鏀诲嚮鎴?DDIM inversion 澶辫触, 涔熶細淇濈暀璇婃柇 records銆乻ummary銆乪nvironment report 鍜?manifest銆?
3. helper 宸叉寜 aligned image 璺緞缁戝畾鍓嶅簭 `prompt_id` 涓?`prompt_text`, 姣忓紶 source image 浣跨敤瀵瑰簲鐪熷疄 prompt 鎵ц鍐嶆墿鏁ｆ敾鍑汇€?
4. helper 宸叉柊澧?`formal_attack_detection_records.jsonl`, 灏嗙湡瀹?attacked image 缁撴灉鎺ュ洖 attack matrix 鍏煎 schema, 骞跺鐢?fixed-FPR threshold 涓?rescue boundary 鐢熸垚姝ｅ紡妫€娴嬭褰曘€?
5. DDIM inversion 璺緞宸叉敼涓轰弗鏍?`DDIMInverseScheduler` inversion + `DDIMScheduler` reconstruction, 榛樿 attacker model 涓?`runwayml/stable-diffusion-v1-5`; 鑻ョ粍浠朵笉鍙敤鎴栧悗绔け璐? 浼氳褰?unsupported 鑰屼笉鏄吉閫犵粨鏋溿€?

## external_baseline_governed_ingestion

| item | value |
| --- | --- |
| construction_unit_name | `external_baseline_comparison` |
| phase_status | `official_source_cached_result_import_ready` |
| executor | `codex_agent` |
| execution_date | `2026-06-21` |
| input_manifest | `outputs/attack_matrix/manifest.local.json`; `outputs/threshold_calibration/threshold_degeneracy_report.json`; `external_baseline/source_registry.json` |
| expected_output_manifest | `outputs/external_baseline_comparison/manifest.local.json` |
| expected_outputs | `outputs/external_baseline_comparison/baseline_observations.jsonl`; `outputs/external_baseline_comparison/baseline_result_records.jsonl`; `outputs/external_baseline_comparison/baseline_metrics.csv`; `outputs/external_baseline_comparison/baseline_comparison_table.csv`; `outputs/external_baseline_comparison/baseline_runtime_report.json`; `outputs/external_baseline_comparison/manifest.local.json` |
| blocking_items | 澶栭儴 baseline 鎸囨爣缁撴灉灏氭湭澶嶇幇鎴栧鍏? 鍥犳 `baseline_results_ready=false` 涓?`supports_paper_claim=false`銆?|
| fallback_path | 鑻ュ畼鏂规簮鐮佹棤娉曠洿鎺ヨ繍琛? 鍙粠鍙楁不鐞嗘潵婧愬鍏?`baseline_result_records.jsonl`, 浣嗗繀椤讳繚鐣欐潵婧愯矾寰勩€佹潵婧愭憳瑕併€佸叡鍚屽崗璁敭鍜屾寚鏍囧瓧娈点€?|
| invariants | 绗笁鏂规簮鐮佷粛淇濈暀鍦ㄨ蹇界暐鐨?`external_baseline/` 缂撳瓨涓? 鏈」鐩彧鎻愪氦 adapter銆佹潵婧愮櫥璁拌鍙栭€昏緫銆佸鍏ヨ褰?schema銆佸姣旇〃閲嶅缓鑴氭湰鍜岃交閲忔祴璇曘€?|

### external baseline 鎺ュ叆鎺ㄨ繘鍐呭

1. `experiments/baselines/adapters.py` 宸叉墿灞曚负鏀寔瀹樻柟婧愮爜鐧昏 overlay 鍜屽彈娌荤悊 `baseline_result_records.jsonl` 瀵煎叆銆?2. `scripts/write_external_baseline_comparison_outputs.py` 宸叉敮鎸佽鍙?`external_baseline/source_registry.json`, 灏嗘湰鍦扮紦瀛樻簮鐮佺姸鎬佸啓鍏ヨ繍琛屾姤鍛? 骞惰緭鍑鸿鑼冨寲鐨?`baseline_result_records.jsonl`銆?3. 褰撳墠鏈湴婧愮爜缂撳瓨涓?8 涓閮?baseline 鍧囧彲琚櫥璁颁负 `baseline_official_code_ready=true`, 浣嗗皻鏃犲叡鍚屽崗璁笅鐨勫鐜板疄娴嬬粨鏋滄垨鍙楁不鐞嗗鍏ョ粨鏋溿€?4. 宸茶ˉ鍏呰交閲忔祴璇? 瑕嗙洊缂哄け缁撴灉鏃剁殑 unsupported 杈圭晫, 浠ュ強鍗曚釜鍙楁不鐞嗗鍏ョ粨鏋滆繘鍏ヨ娴嬭褰曘€佽仛鍚堟寚鏍囧拰鍏卞悓鍗忚瀵规瘮琛ㄧ殑閾捐矾銆?
### external baseline 褰撳墠浜х墿鎽樿

1. `baseline_runtime_report.json` 鏄剧ず `baseline_source_registry_ready=true`, `official_source_ready_count=8`, `imported_baseline_result_count=0`, `baseline_result_ready_count=0`, `baseline_results_ready=false`銆?2. `baseline_metrics.csv` 浠嶄繚鎸佹墍鏈?baseline 鐨勭粨鏋滀负 unsupported, 璇ョ姸鎬佹槸姝ｇ‘杈圭晫, 琛ㄧず瀹樻柟婧愮爜宸茬紦瀛樹絾灏氭湭褰㈡垚鍏卞悓鍗忚鎸囨爣銆?3. 涓嬩竴姝ュ簲閫夋嫨涓昏〃 baseline 浼樺厛绾? 鍏堝 Tree-Ring銆丟aussian Shading銆丼hallow Diffuse 鍜?T2SMark 寤虹珛鍙楁不鐞嗗鐜版垨瀵煎叆璁板綍, 鍐嶆墿灞曡ˉ鍏呰〃 baseline銆?
### external baseline 楠岃瘉缁撴灉

| command | result |
| --- | --- |
| `python -m py_compile experiments/baselines/adapters.py experiments/baselines/__init__.py scripts/write_external_baseline_comparison_outputs.py tests/functional/test_external_baseline_comparison.py` | pass |
| `pytest tests/functional/test_external_baseline_comparison.py tests/functional/test_external_baseline_source_registry.py -q` | pass, 5 passed |
| `python scripts/write_external_baseline_comparison_outputs.py` | pass, `official_source_ready_count=8`, `baseline_results_ready=false` |


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
| blocking_items | Tree-Ring、Gaussian Shading 和 Shallow Diffuse 仍需要真实 SD3.5 Medium GPU adapter 实装; 当前只提供可审计命令契约。 |
| fallback_path | 若官方源码暂不能直接适配 SD3.5 Medium, 只能保留 `contract-only` 诊断或导入受治理结果, 不得声明论文级 baseline 结论。 |
| invariants | 官方源码快照位于 `external_baseline/*/*/source/` 且不由 git 跟踪; 项目维护 adapter、命令计划脚本、执行脚本和证据校验脚本必须接受 harness 审计。 |

### external baseline 并入方法修正

1. 根 `.gitignore` 不再忽略整个 `external_baseline/`, 改由 `external_baseline/.gitignore` 仅忽略第三方 `source/` 与 adapter 临时 `artifacts/` 子树。
2. `tools/harness/lib/file_scanner.py` 已改为扫描 `external_baseline/` 中的 adapter、README 和登记文件, 但跳过第三方源码快照。
3. 新增 `experiments/baselines/command_adapter.py`、`command_plan.py`、`observation_io.py` 和 `evidence_validator.py`, 形成 command plan、execution、observation 和 evidence 的统一入口。
4. 新增 `scripts/build_external_baseline_command_plan.py`、`scripts/run_external_baseline_command_plan.py` 和 `scripts/validate_external_baseline_evidence.py`, 所有仓库命令输出默认写入 `outputs/`。
5. 主表 baseline 已新增项目维护 adapter 路径。T2SMark adapter 可读取官方 `results.json`; Tree-Ring、Gaussian Shading 和 Shallow Diffuse 当前提供 `contract-only` 诊断入口, 真实指标仍需补齐 SD3.5 Medium GPU 运行路径。
6. `external_baseline/source_registry.json` 已补充 `adapter_path`、`adapter_status`、`model_alignment_status` 和 `official_source_tracked` 字段。

### 当前边界

1. 本次变更建立外部 baseline 的实施流程, 不伪造外部 baseline 真实指标。
2. `contract-only` 只证明命令编排、adapter 落盘和 harness 边界可用。
3. 论文级对比必须有 `formal_result_claim=true`、真实 `evidence_paths`、官方源码 commit、运行日志和可重建 observation 或受治理结果记录。
