from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pm4py

from functions import *

# ============================================================
# CONSTANTS
# ============================================================

# Project root (where main_script.py and functions.py are located).
PROJECT_ROOT = Path("YOUR_ROOT_PROJECT_PATH")
XES_FILE_NAME = "YOUR_LOG.xes"
XES_LOG_PATH = PROJECT_ROOT / "YOUR_LOG_FOLDER" / XES_FILE_NAME

# Where the modified cortado-core repo is cloned.
CORTADO_CORE_REPO_DIR = PROJECT_ROOT / "ROOT_REPO_FOLDER" / "cortado-core"

# Output folder.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Discovery algorithm (pm4py-core): "inductive" or "heuristics".
DISCOVERY_ALGO = "inductive"

# Unfolding variants (cortado-core): 1 baseline, 2 ERV optimized, 3 ERV[|>h].
UNFOLDING_VARIANTS = [3]
ACTIVITY_KEY = "concept:name"
CASE_ID_KEY = "concept:name"

# Sampling parameters.
SAMPLE_TRACES = 1000

# True = stratified, False = random
STRATIFIED_SAMPLE = True
SAMPLE_BUCKETS = 19
SAMPLE_SEED = 42


def main() -> None:
    # Timestamped output folder.
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = OUTPUTS_DIR / run_id
    ensure_dir(out_dir)
    print(f"Run: {run_id}\n")
    print(f"Log: {XES_LOG_PATH}\n")
    print(f"Cortado-core repo: {CORTADO_CORE_REPO_DIR}\n")
    print(f"Output dir: {out_dir}\n")

    # =========================
    # 1) Log import and stratified sampling.
    # =========================

    print("\n[1] Loading event log (XES)")
    log = load_event_log_pm4py(XES_LOG_PATH)
    #print(f"[DEBUG] Traces in the log: {len(log)}")
    #print("[DEBUG] SAMPLE_TRACES = ", SAMPLE_TRACES, type(SAMPLE_TRACES))
    #print("[DEBUG] STRATIFIED_SAMPLE = ", STRATIFIED_SAMPLE)
    #print("[DEBUG]", log_stats(log))
    if SAMPLE_TRACES is not None:
        print(f"[Sampling] Using {SAMPLE_TRACES} traces ("
             f"{'stratified' if STRATIFIED_SAMPLE else 'random'})")
    if is_pandas_dataframe(log):
        if STRATIFIED_SAMPLE:
            log = stratified_case_sample_df(log,
                                            sample_size=SAMPLE_TRACES,
                                            n_buckets=SAMPLE_BUCKETS,
                                            seed=SAMPLE_SEED)
        else:
            log = random_case_sample_df(log,
                                        sample_size=SAMPLE_TRACES,
                                        seed=SAMPLE_SEED)
    else:
        raise RuntimeError(f"A DataFrame was expected, if the event log is of" + 
                           f" type EventLog, the sampling function needs to " +
                           f"be adjusted\n")
    case_col = get_case_id_column(log)
    log_el = dataframe_to_eventlog(log, case_id_col=case_col)
    #print(f"[DEBUG] {log_stats(log)}")
    #print(f"[DEBUG] Number of traces in log after sampling: {len(log)}")
    #print(f"[DEBUG] traces materialized: {len([t for t in log])}")
    #print(f"[DEBUG] Cases in DF: {log[get_case_id_column(log)].nunique()}")
    #print(f"[DEBUG] Traces in EventLog: {len(log_el)}")
    export_sampled_xes(log, f"{out_dir}/sample_{XES_FILE_NAME}")
    sampled_case_ids_path = out_dir / "sampled_case_ids.json"
    save_sampled_case_ids(
        df=log,
        out_path=sampled_case_ids_path,
        case_id_col=case_col
    )
    print(f"[SAVE] Sampled case IDs saved to: {sampled_case_ids_path}\n")
    sampling_metadata_path = out_dir / "sampling_metadata.json"
    save_sampling_metadata(
        sampling_metadata_path,
        {
            "sample_traces": SAMPLE_TRACES,
            "stratified": STRATIFIED_SAMPLE,
            "buckets": SAMPLE_BUCKETS,
            "seed": SAMPLE_SEED,
            "case_id_column": case_col,
            "n_events": len(log),
            "n_cases": log[case_col].nunique(),
        }
    )
    print(f"[SAVE] Sampling metadata saved to: {sampling_metadata_path}\n")

    # =========================
    # 2) Discovery Petri net + export PNML.
    # =========================

    print(f"\n[2] Discovery Petri net ({DISCOVERY_ALGO}) + export PNML\n")
    net, im, fm = discover_petri_net(log, algo=DISCOVERY_ALGO)
    pnml_path = out_dir / "discovered_model.pnml"
    export_pnml(net, im, fm, pnml_path)
    print(f"    PNML salvato: {pnml_path}\n")
    net2, im2, fm2 = pm4py.read_pnml(str(pnml_path))
    #print("[DEBUG PNML] im2 = ", im2)
    #print("[DEBUG PNML] fm2 = ", fm2)

    # =========================
    # 3) Conformance checking.
    # =========================

    results = []


    # 3.1 Standard alignments (pm4py).

    print("\n[3.1] Alignments standard (pm4py-core)")
    details_path = out_dir / "details_pm4py_alignments.json"
    r_std, _ = conformance_alignments_standard_pm4py(log,
                                                     net,
                                                     im,
                                                     fm,
                                                     save_details_to=details_path)
    results.append(r_std)
    print(f"    -> conformant={r_std.conformant_pct:.2f}% " + 
          f"wall={r_std.wall_time_s:.3f}s peakRSS={r_std.peak_rss_mb:.1f}MB\n")


    # 3.2) Variant grouping for Cortado (log compression).

    sampled_xes_path = out_dir / f"sample_{XES_FILE_NAME}" 
    variant_xes_path = out_dir / f"variants_{XES_FILE_NAME}"
    variant_mapping_path = out_dir / "variants_mapping.json"
    export_variant_compressed_xes(
        original_xes=sampled_xes_path,
        out_xes=variant_xes_path,
        out_mapping_json=variant_mapping_path,
        activity_key=ACTIVITY_KEY,
        case_id_key=CASE_ID_KEY 
    )
    print(f"[SAVE] Variant-compressed XES: {variant_xes_path}\n")
    print(f"[SAVE] Variant mapping JSON: {variant_mapping_path}\n")
    print("\n[3.2] Unfolding-based alignments (cortado-core)")
    for v in UNFOLDING_VARIANTS:
        r = conformance_unfolding_cortado_cli(
            cortado_core_repo_dir=CORTADO_CORE_REPO_DIR,
            out_dir=out_dir,
            xes_path=variant_xes_path,
            pnml_path=pnml_path,
            variant=v,
            run_tag="run_variants"
        )
        results.append(r)
        avg_per_trace = ((r.per_trace_time_sum_s / r.n_traces) 
                         if (r.per_trace_time_sum_s is not None and r.n_traces) 
                         else None)
        time_str = (f"{avg_per_trace:.4f}s" 
                    if (avg_per_trace is not None) 
                    else "n/a")
        print(f"\t{r.method} v{v}: conformant={r.conformant_pct:.2f}%  "
              f"wall_time={r.wall_time_s:.2f}s  time_per_trace={time_str}  "
              f"peak_rss={r.peak_rss_mb:.1f}MB\n")
        
        # Per-case expansion and results saving.
        cortado_raw_csv = out_dir / f"unfolding_v{v}_run_variants_raw.csv"
        expanded_csv = out_dir / f"unfolding_v{v}_run_variants_expanded_per_case.csv"
        expand_cortado_variant_csv_to_cases(
            cortado_csv_no_header=cortado_raw_csv,
            mapping_json=variant_mapping_path,
            out_csv=expanded_csv
        )
        print(f"[SAVE] Expanded per-case CSV: {expanded_csv}\n")
        weighted = weighted_conformance_from_cortado_variants(
            cortado_csv_no_header=cortado_raw_csv,
            mapping_json=variant_mapping_path
        )
    print(
        f"[INFO] Weighted strict conformance: " + 
        f"{weighted['conformant_pct_weighted']:.9f}% "
        f"(cases={int(weighted['total_cases'])})\n"
    )

    # =========================
    # 4) Comparison and saving.
    # =========================

    print("\n[4] Saving summary")
    save_summary(out_dir, results)

    # =========================
    # 5) Report comparing PM4Py vs Cortado.
    # =========================

    cortado_raw_csvs = []
    for v in UNFOLDING_VARIANTS:
        cortado_raw_csvs.append(out_dir / 
                                f"unfolding_v{v}_run_variants_raw.csv")
    comparison_csv = out_dir / "comparison_pm4py_vs_cortado.csv"
    df_cmp = build_alignment_comparison_report(
        pm4py_details_json=details_path,
        cortado_csv_paths=cortado_raw_csvs,
        out_csv_path=comparison_csv
    )
    print(f"[SAVE] Comparison report saved to: {comparison_csv}")
    print(f"[DEBUG] First rows comparison:\n"+
          f"{df_cmp.head(5).to_string(index=False)}")

    # =========================
    # Comparable report building (1:1) (PM4Py vs expanded Cortado)
    # =========================

    sampled_xes_path = out_dir / f"sample_{XES_FILE_NAME}"
    details_path = out_dir / "details_pm4py_alignments.json"
    cortado_expanded = out_dir / "unfolding_v3_run_variants_expanded_per_case.csv"
    comparison_per_case = out_dir / "comparison_pm4py_vs_cortado_per_case.csv"
    df_case = build_comparison_report_per_case(
        sampled_xes_path=sampled_xes_path,
        pm4py_details_json=details_path,
        cortado_expanded_per_case_csv=cortado_expanded,
        out_csv_path=comparison_per_case,
        case_id_key="concept:name"
    )
    print(f"[SAVE] Comparison per-case report: {comparison_per_case}")
    print((df_case[["case_id", 
                    "pm4py_cost", 
                    "cortado_cost"]]).head(10).to_string(index=False))



if __name__ == "__main__":
    main()
