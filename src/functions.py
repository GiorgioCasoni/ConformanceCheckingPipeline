from __future__ import annotations
import json
import os
import sys
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import psutil
import random
import math
import numpy as np
from collections import defaultdict
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter

# =========================
# Data structures
# =========================

@dataclass
class MethodResult:
    """
    Container for the aggregated outcome of one conformance-checking method run.

    This data structure stores the main metrics collected during an experiment,
    including the method identifier, the number of evaluated traces, the 
    percentage of strictly conformant traces, total wall-clock execution time, 
    optional cumulative per-trace runtime, peak memory usage, and optional notes 
    or paths to raw output files. It is used as the common result format for 
    both PM4Py-based and Cortado-based runs.
    """
    method: str
    n_traces: int
    conformant_pct: float
    wall_time_s: float
    per_trace_time_sum_s: Optional[float]
    peak_rss_mb: float
    notes: Optional[str] = None
    raw_output_path: Optional[str] = None

# =========================
# Utilities
# =========================

def ensure_dir(p: Path) -> None:
    """
    Create a directory if it does not already exist.

    This utility ensures that the target path is available before saving any 
    output artifacts such as logs, summaries, models, mappings, or comparison 
    reports. The directory and all missing parent directories are created 
    safely.
    """
    p.mkdir(parents=True, exist_ok=True)


def bytes_to_mb(x: int) -> float:
    return x / (1024.0 * 1024.0)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """
    Execute an external command and return its exit code.

    This function is used to invoke the modified Cortado CLI from Python while
    optionally setting the working directory. It is intended for experiment
    automation, where only the command return code is needed to detect success 
    or failure.
    """
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True
    )
    return completed.returncode


def fmt10(x):
    """
    Format a numeric value using ten decimal digits.

    If the input value is missing, an empty string is returned instead. This 
    helper is mainly used when exporting comparison reports in order to preserve 
    numerical precision for fitness and cost-related fields.
    """
    return f"{float(x):.10f}" if pd.notna(x) else ""


class PeakRSSSampler:
    """
    Background sampler for approximate peak resident memory usage.

    This class periodically inspects the RSS of the current Python process while 
    a computation is running and stores the maximum observed value. It is used 
    as a lightweight approximation of peak memory consumption during alignment
    experiments.
    """
    def __init__(self, interval_s: float = 0.02) -> None:
        """
    Initialize the RSS sampler.

    Args:
        interval_s: Time in seconds between two consecutive memory samples.
    """
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_bytes: int = 0

    def start(self) -> None:
        """
    Start sampling the current process memory in a background thread.

    The sampler keeps running until `stop()` is called and updates the stored 
    peak RSS whenever a larger value is observed.
    """
        proc = psutil.Process(os.getpid())

        def _run() -> None:
            while not self._stop.is_set():
                try:
                    rss = proc.memory_info().rss
                    if rss > self.peak_rss_bytes:
                        self.peak_rss_bytes = rss
                except Exception:
                    pass
                time.sleep(self.interval_s)

        self._stop.clear()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
    Stop the background memory sampler and wait briefly for the thread to finish.

    After this call, the `peak_rss_bytes` attribute contains the highest RSS 
    value observed since the sampler was started.
    """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# -------------------------
# Section 1: log import (pm4py-core)
# -------------------------

def load_event_log_pm4py(xes_path: Path):
    """
    Load an event log from a XES file using PM4Py.

    This function is the standard entry point for importing the input log used 
    in the experiments. Depending on the PM4Py version and settings, the 
    returned object may be a pandas DataFrame or an EventLog-like structure.
    """
    return pm4py.read_xes(str(xes_path))


# -------------------------
# Section 2: discovery Petri net (pm4py-core)
# -------------------------

def discover_petri_net(log, algo: str = "inductive"):
    """
    Discover a Petri net from an event log using the selected PM4Py algorithm.

    Supported discovery modes include inductive mining and heuristics mining. 
    The function returns the discovered net together with its initial and final
    markings, so that the model can be exported and later reused for conformance
    checking.
    """
    if algo == "inductive":
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.1)
        return net, im, fm
    elif algo == "heuristics":
        hn = pm4py.discover_heuristics_net(log)
        net, im, fm = pm4py.convert_to_petri_net(hn)
        return net, im, fm
    else:
        raise ValueError(f"Unsupported discovery algo: {algo}")


def export_pnml(net, im, fm, pnml_path: Path) -> None:
    """
    Export a discovered Petri net to PNML format.

    This function writes the Petri net and its markings to disk so that the 
    model can be reloaded later or passed to Cortado for unfolding-based 
    alignments.
    """
    pm4py.write_pnml(net, im, fm, str(pnml_path))


# -------------------------
# Section 3.1: standard alignments (pm4py-core)
# -------------------------

def conformance_alignments_standard_pm4py(log,
                                          net,
                                          im,
                                          fm,
                                          save_details_to: Optional[Path] = None
                                          ) -> Tuple[MethodResult,
                                                     Optional[List
                                                              [Dict
                                                               [str, Any]]]]:
    """
    Compute standard alignments with PM4Py and summarize their performance.

    The function runs PM4Py alignment diagnostics on the given log and model,
    measures wall-clock time and approximate peak memory, reconstructs an exact
    fitness value when `bwc` is available, and computes strict conformance using
    the criterion `cost == 0`. Optionally, the raw alignment details are also 
    saved to disk for later inspection and comparison.
    """
    sampler = PeakRSSSampler()
    sampler.start()
    t0 = time.perf_counter()
    alignments = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
    wall = time.perf_counter() - t0
    sampler.stop()
    n = len(alignments)
    conformant_strict = 0
    per_trace_time = 0.0
    any_time = False

    # For debug and analysis: reconstructs "exact" fitness when we have the bwc.
    # (Avoids rounding to 1.0 when cost is small and bwc much larger).
    for a in alignments:
        cost = a.get("cost",
                     a.get("alignment_cost",
                           a.get("alignment_costs",
                                 None)))
        bwc = a.get("bwc", None)

        # STRICT: conformant ⇔ cost == 0
        if cost is not None and float(cost) == 0.0:
            conformant_strict += 1

        # Adds a "fitness_exact", if possible.
        # Formula: 1 - cost/bwc (if bwc>0)
        try:
            if cost is not None and bwc is not None and float(bwc) > 0:
                a["fitness_exact"] = 1.0 - (float(cost) / float(bwc))
        except Exception:
            pass
        if "time" in a:
            per_trace_time += float(a["time"])
            any_time = True
        elif "runtime" in a:
            per_trace_time += float(a["runtime"])
            any_time = True

        if "time" in a:
            per_trace_time += float(a["time"])
            any_time = True
        elif "runtime" in a:
            per_trace_time += float(a["runtime"])
            any_time = True

    res = MethodResult(
        method="pm4py_standard_alignments",
        n_traces=n,
        conformant_pct=(conformant_strict / n * 100.0) if n else 0.0,
        wall_time_s=wall,
        per_trace_time_sum_s=per_trace_time if any_time else None,
        peak_rss_mb=bytes_to_mb(sampler.peak_rss_bytes),
        notes=(f"conformant (strict): cost==0; fitness_exact added when " +
               f"bwc available")
    )

    if save_details_to is not None:
        save_details_to.write_text(json.dumps(alignments,
                                              indent=2,
                                              ensure_ascii=False),
                                              encoding="utf-8")
    print(f"[DEBUG] alignments returned: {len(alignments)}")
    return res, alignments


# -------------------------
# Section 3.2-3.4: unfolding-based alignments (cortado-core CLI)
# -------------------------

def parse_cortado_unfolding_csv_no_header(csv_path: Path) -> pd.DataFrame:
    """
    The product of Cortado unfoldings is a CSV without an header, this function
    maps columns.
    Robust mapping:
      col0: run_label (ex. "Variant.BASELINE")
      col1: idx (1..N)
      col2: trace_len
      col3: time_1
      col4: time_2
      col5: size_1
      col6: size_2
      col7: cost
    """
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 8:
        raise ValueError(f"CSV Cortado inatteso: colonne={df.shape[1]} (<8). " +
                         f"File: {csv_path}")
    
    # Should it have more than 8 columns, takes the first 8 for compatibility.
    # WARNING: it may cause loss of data.
    df = df.iloc[:, :8].copy()
    df.columns = ["run_label",
                  "idx",
                  "trace_len",
                  "time_1",
                  "time_2",
                  "size_1",
                  "size_2",
                  "cost"]
    
    # cast safe
    for c in ["idx", "trace_len", "size_1", "size_2", "cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["time_1", "time_2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def conformance_unfolding_cortado_cli(cortado_core_repo_dir: Path,
                                      out_dir: Path,
                                      xes_path: Path,
                                      pnml_path: Path,
                                      variant: int,
                                      run_tag: str) -> MethodResult:
    """
    Executes cortado-core (cortado_core/alignments/unfolding/test.py) and reads
    the produced CSV file.

    This function prepares a temporary working directory, copies the input XES 
    log and PNML model, invokes the selected unfolding variant, parses the 
    produced CSV output, and returns a `MethodResult` containing strict 
    conformance, runtime, and approximate peak memory. The raw Cortado CSV is 
    also copied into the main output directory for traceability.

    Variants (from the README of the fork ariba-work/cortado-core):
        - v=0 baseline,
        - v=1 ERV optimized,
        - v=2 ERV[|>h]
    """
    unfolding_dir = cortado_core_repo_dir / "cortado_core" / "alignments" / "unfolding"
    test_py = unfolding_dir / "test.py"

    if not test_py.exists():
        raise FileNotFoundError(
            f"{test_py} not found. Did you clone ariba-work/cortado-core in: " +
            f"{cortado_core_repo_dir} ?"
        )

    # Work dir
    work_dir = out_dir / "work_cortado"
    ensure_dir(work_dir / "results")
    log_name = f"log_{run_tag}.xes"
    model_name = f"model_{run_tag}.pnml"
    work_log = work_dir / log_name
    work_model = work_dir / model_name
    shutil.copy2(xes_path, work_log)
    shutil.copy2(pnml_path, work_model)
    cmd = [
        sys.executable,
        "-u",
        str(test_py),
        "unfolding-based-alignments",
        "-p", str(work_dir),
        "-l", log_name,
        "-m", model_name,
        "-v", str(variant),
    ]
    sampler = PeakRSSSampler()
    sampler.start()
    t0 = time.perf_counter()
    #rc, out, err = run_cmd(cmd, cwd=unfolding_dir)
    rc = run_cmd(cmd, cwd=unfolding_dir)
    wall = time.perf_counter() - t0
    sampler.stop()
    if rc != 0:
        raise RuntimeError(
            f"Errore while executing cortado-core (variant={variant}).\n"
            f"CMD: {' '.join(cmd)}"
        )
    produced_csv = work_dir / "results" / f"{model_name}.csv"
    if not produced_csv.exists():
        raise FileNotFoundError(f"Expected CSV not found: " + 
                                f"{produced_csv}")
    raw_out = out_dir / f"unfolding_v{variant}_{run_tag}_raw.csv"
    shutil.copy2(produced_csv, raw_out)
    df = parse_cortado_unfolding_csv_no_header(raw_out)
    n = len(df)
    # STRICT: conformant ⇔ cost == 0
    conformant = int((df["cost"] == 0).sum()) if n else 0
    # Choice of the per-trace time: uses time_2 if it exists, time_1 otherwise.
    if n and df["time_2"].notna().any():
        per_trace_time_sum = float(df["time_2"].fillna(0).sum())
    elif n and df["time_1"].notna().any():
        per_trace_time_sum = float(df["time_1"].fillna(0).sum())
    else:
        per_trace_time_sum = None
    label = {
        0: "cortado_unfolding_baseline_v0",
        1: "cortado_unfolding_ERV_optimized_v1",
        2: "cortado_unfolding_ERVh_v2",
    }.get(variant, f"cortado_unfolding_v{variant}")

    return MethodResult(
        method=label,
        n_traces=n,
        conformant_pct=(conformant / n * 100.0) if n else 0.0,
        wall_time_s=wall,
        per_trace_time_sum_s=per_trace_time_sum,
        peak_rss_mb=bytes_to_mb(sampler.peak_rss_bytes),
        raw_output_path=str(raw_out)
    )


# -------------------------
# Section 4: save + print
# -------------------------

def save_summary(out_dir: Path,
                 results: List[MethodResult]) -> Tuple[Path, Path]:
    """
    Save the aggregated experimental results in both CSV and JSON format.

    The function converts a list of `MethodResult` objects into a tabular 
    summary, writes the summary to disk, prints a compact console overview, and 
    returns the paths of the generated files.
    """
    ensure_dir(out_dir)
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps([asdict(r) for r in results],
                                    indent=2,
                                    ensure_ascii=False),
                        encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(df[["method", "n_traces", "conformant_pct", "wall_time_s", 
              "peak_rss_mb", "raw_output_path"]].to_string(index=False))
    print(f"\nSalvati: {csv_path} , {json_path}")

    return csv_path, json_path


def get_trace_length(trace) -> int:
    return len(trace)


def stratified_trace_sample(log,
                            sample_size: int,
                            n_buckets: int = 5,
                            seed: int = 42) -> EventLog:
    """
    Sample traces from an EventLog using stratification by trace length.

    The function groups traces into buckets according to their length, samples 
    from each bucket to preserve heterogeneity, and builds a new EventLog 
    containing copied traces and preserved trace attributes. It is intended for 
    robust subsampling of event logs represented as trace collections.
    """
    rng = random.Random(seed)

    # Materializes all traces one time.
    traces = [t for t in log]
    n = len(traces)
    if sample_size is None or sample_size >= n:
        return log
    idx_len: List[Tuple[int, int]] = [(i, len(traces[i])) for i in range(n)]
    idx_len.sort(key=lambda x: x[1])
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for pos, (idx, _) in enumerate(idx_len):
        b = min(n_buckets - 1, math.floor(pos / n * n_buckets))
        buckets[b].append(idx)
    per_bucket = sample_size // n_buckets
    remainder = sample_size % n_buckets
    chosen = []
    chosen_set = set()
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        k = per_bucket + (1 if i < remainder else 0)
        k = min(k, len(bucket))
        pick = rng.sample(bucket, k)
        for idx in pick:
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
    if len(chosen) < sample_size:
        remaining = [i for i in range(n) if i not in chosen_set]
        chosen.extend(rng.sample(remaining, sample_size - len(chosen)))

    # Builds a new EventLog with copies of the original traces.
    sampled_log = EventLog()
    for idx in chosen:
        original_trace = traces[idx]
        new_trace = Trace([dict(ev) for ev in original_trace])
        try:
            new_trace.attributes = dict(getattr(original_trace,
                                                "attributes",
                                                {}))
        except Exception:
            pass
        sampled_log.append(new_trace)

    # Copies log attributes (optional).
    try:
        sampled_log.attributes = dict(getattr(log,
                                              "attributes",
                                              {}))
    except Exception:
        pass

    return sampled_log


def random_trace_sample(log, sample_size: int, seed: int = 42):
    """
    Sample traces uniformly at random from an EventLog.

    A new EventLog is returned, containing copied traces selected without
    replacement. Trace and log attributes are preserved whenever possible.
    """
    rng = random.Random(seed)
    traces = [t for t in log]
    n = len(traces)
    if sample_size is None or sample_size >= n:
        return log
    chosen = rng.sample(range(n), sample_size)
    sampled_log = EventLog()
    for idx in chosen:
        original_trace = traces[idx]
        new_trace = Trace([dict(ev) for ev in original_trace])
        try:
            new_trace.attributes = dict(getattr(original_trace,
                                                "attributes",
                                                {}))
        except Exception:
            pass
        sampled_log.append(new_trace)
    try:
        sampled_log.attributes = dict(getattr(log,
                                              "attributes",
                                              {}))
    except Exception:
        pass

    return sampled_log

def export_sampled_xes(log, out_xes_path: Path) -> None:
    pm4py.write_xes(log, str(out_xes_path))


def is_pandas_dataframe(obj) -> bool:
    return isinstance(obj, pd.DataFrame)


def get_case_id_column(df: pd.DataFrame) -> str:
    """
    Infer the case identifier column name from a pandas DataFrame log.

    The function searches a small set of common case-id column names and returns
    the first match. It raises an error when no supported case identifier column 
    is found.
    """
    candidates = ["case:concept:name", "case_id", "case", "CaseID", "caseid"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Non trovo una colonna case-id. Colonne disponibili: " +
                     f"{list(df.columns)[:30]} ...")


def get_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Infer the timestamp column name from a pandas DataFrame log.

    The function searches a small set of common timestamp column names and 
    returns the first match, or `None` if no timestamp column is available.
    """
    candidates = ["time:timestamp", "timestamp", "time", "TimeStamp"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def stratified_case_sample_df(df: pd.DataFrame,
                              sample_size: int,
                              n_buckets: int = 5,
                              seed: int = 42,
                              case_id_col: Optional[str] = None
                              ) -> pd.DataFrame:
    """
    Sample cases from a DataFrame log using stratification by trace length.

    Cases are first ordered by their number of events, then partitioned into
    buckets, and finally sampled approximately uniformly across buckets. This 
    helps preserve short, medium, and long traces within the sampled log.
    """
    if case_id_col is None:
        case_id_col = get_case_id_column(df)

    # Number of traces.
    case_sizes = df.groupby(case_id_col).size().rename("trace_len")
    n_cases = case_sizes.shape[0]
    if sample_size is None or sample_size >= n_cases:
        return df
    rng = random.Random(seed)

    # Sorts cases by trace_len.
    case_sizes_sorted = case_sizes.sort_values()
    case_ids = case_sizes_sorted.index.to_list()

    # Creates buckets.
    buckets = [[] for _ in range(n_buckets)]
    for pos, cid in enumerate(case_ids):
        b = min(n_buckets - 1, int(pos / n_cases * n_buckets))
        buckets[b].append(cid)
    per_bucket = sample_size // n_buckets
    remainder = sample_size % n_buckets
    chosen = []
    chosen_set = set()
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        k = per_bucket + (1 if i < remainder else 0)
        k = min(k, len(bucket))
        pick = rng.sample(bucket, k)
        for cid in pick:
            if cid not in chosen_set:
                chosen.append(cid)
                chosen_set.add(cid)

    # "Fills" the sample if necessary.
    if len(chosen) < sample_size:
        remaining = [cid for cid in case_ids if cid not in chosen_set]
        chosen.extend(rng.sample(remaining, sample_size - len(chosen)))
    sampled_df = df[df[case_id_col].isin(chosen)].copy()
    
    return sampled_df


def random_case_sample_df(df: pd.DataFrame,
                          sample_size: int,
                          seed: int = 42,
                          case_id_col: Optional[str] = None) -> pd.DataFrame:
    """
    Sample cases uniformly at random from a DataFrame log.

    The function selects a subset of case identifiers without replacement and
    returns the filtered DataFrame containing all events belonging to the chosen
    cases.
    """
    if case_id_col is None:
        case_id_col = get_case_id_column(df)
    case_ids = df[case_id_col].unique().tolist()
    n_cases = len(case_ids)
    if sample_size is None or sample_size >= n_cases:
        return df
    rng = random.Random(seed)
    chosen = rng.sample(case_ids, sample_size)
    return df[df[case_id_col].isin(chosen)].copy()


def log_stats(df_or_log) -> str:
    """
    Returns a compact textual summary of a log.

    Prints statistics:
    - if the log is a DataFrame: #events, #cases, min/mean/max trace length
    - if the log is a EventLog: #traces and min/mean/max trace length
    """
    if is_pandas_dataframe(df_or_log):
        df = df_or_log
        case_id = get_case_id_column(df)
        sizes = df.groupby(case_id).size()
        return (
            f"DataFrame log | events={len(df)} | cases={sizes.shape[0]} | "
            f"trace_len(min/mean/max)={sizes.min()}/{sizes.mean():.2f}/" +
            f"{sizes.max()}"
        )
    else:
        log = df_or_log
        lengths = [len(t) for t in log]
        return (f"EventLog | traces={len(lengths)} | " +
                f"trace_len(min/mean/max)={min(lengths)}/" +
                f"{(sum(lengths)/len(lengths)):.2f}/{max(lengths)}")


def dataframe_to_eventlog(df: pd.DataFrame,
                          case_id_col: str = "case:concept:name"):
    """
    Convert a pandas DataFrame log into a PM4Py EventLog.

    The function uses the detected or provided case-id column to group events 
    into traces and produces an EventLog representation suitable for APIs that 
    operate on trace-based logs.
    """
    if case_id_col not in df.columns:
        case_id_col = get_case_id_column(df)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 
                  case_id_col}
    event_log = log_converter.apply(df,
                                    parameters=parameters,
                                    variant=log_converter.Variants.TO_EVENT_LOG)

    return event_log


def save_sampled_case_ids(df: pd.DataFrame,
                          out_path: Path,
                          case_id_col: str) -> None:
    """
    Save the identifiers of the sampled cases to disk.

    This artifact makes the selected sample explicitly reproducible by recording 
    the exact case identifiers included in the experiment.
    """
    case_ids = sorted(df[case_id_col].unique().tolist())
    out_path.write_text(
        json.dumps(case_ids, indent=2),
        encoding="utf-8"
    )


def save_sampling_metadata(out_path: Path,
                           metadata: dict) -> None:
    """
    Saves sampling metadata (seed, buckets).
    """
    out_path.write_text(json.dumps(metadata, indent=2),
                        encoding="utf-8")


def export_eventlog_xes(log_el,
                        out_xes_path: Path) -> None:
    """
    Exports an EventLog pm4py in XES format.
    """
    pm4py.write_xes(log_el, str(out_xes_path))


def build_alignment_comparison_report(pm4py_details_json: Path,
                                      cortado_csv_paths: List[Path],
                                      out_csv_path: Path) -> pd.DataFrame:
    """
    Creates a tabular report for the comparison PM4py vs Cortado.
    Joins by index (1..N) assuming a trace order coherent with the Cortado's 
    CSV.
    """
    # --- PM4Py ---

    details = json.loads(pm4py_details_json.read_text(encoding="utf-8"))
    rows = []
    for i, a in enumerate(details, start=1):
        cost = a.get("cost",
                     a.get("alignment_cost",
                           a.get("alignment_costs", None)))
        bwc = a.get("bwc", None)
        fitness = a.get("fitness", None)
        fitness_exact = a.get("fitness_exact", None)
        if fitness_exact is None:
            try:
                if cost is not None and bwc is not None and float(bwc) > 0:
                    fitness_exact = 1.0 - (float(cost) / float(bwc))
            except Exception:
                fitness_exact = None
        rows.append({
            "idx": i,
            "pm4py_cost": (float(cost) if (cost is not None)
                           else None),
            "pm4py_bwc": (float(bwc) if bwc is not None
                          else None),
            "pm4py_fitness_printed": (float(fitness) if fitness is not None
                                      else None),
            "pm4py_fitness_exact": (float(fitness_exact)
                                    if fitness_exact is not None 
                                    else None),
            "pm4py_conformant_strict": ((float(cost) == 0.0)
                                        if cost is not None
                                        else None)
        })
    df_pm = pd.DataFrame(rows)

    # --- Cortado: a single column for every CSV passed ---

    df_out = df_pm.copy()
    for p in cortado_csv_paths:
        df_c = parse_cortado_unfolding_csv_no_header(p)[["idx", "cost"]].copy()
        col = f"cortado_cost__{p.stem}"
        df_c = df_c.rename(columns={"cost": col})
        df_out = df_out.merge(df_c, on="idx", how="left")
        df_out[f"cortado_conformant_strict__{p.stem}"] = (df_out[col] == 0)
    df_out["pm4py_fitness_printed_10"] = (
        df_out["pm4py_fitness_printed"].apply(fmt10))
    ensure_dir(out_csv_path.parent)
    df_out.to_csv(out_csv_path, index=False)

    return df_out


def build_variants_from_xes(xes_path: Path,
                            activity_key: str = "concept:name",
                            case_id_key: str = "concept:name",):
    """
    Extract and group trace variants from a XES log.

    Each variant is identified by its ordered activity sequence and represented 
    with a stable index, its frequency, the list of original case identifiers, 
    and one example trace. The output is designed to support variant compression 
    before running Cortado.
    Variants:
      [{"variant_idx": 1, "variant_key": tuple(...), "freq": n,
      "case_ids": [...], "example_trace": Trace}, ...]
    """
    log = xes_importer.apply(str(xes_path))
    groups = defaultdict(list)
    example = {}
    for tr in log:
        vkey = tuple(ev.get(activity_key, "") for ev in tr)
        cid = tr.attributes.get(case_id_key, None)
        groups[vkey].append(cid)
        if vkey not in example:
            example[vkey] = tr

    # Orders by descending frequency, lenght of desc and lexicographic key.
    keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), -len(k), k))
    variants = []
    for i, k in enumerate(keys, start=1):
        variants.append({
            "variant_idx": i,
            "variant_key": k,
            "freq": len(groups[k]),
            "case_ids": groups[k],
            "example_trace": example[k],
        })

    return variants


def export_variant_compressed_xes(original_xes: Path,
                                  out_xes: Path,
                                  out_mapping_json: Path,
                                  activity_key: str = "concept:name",
                                  case_id_key: str = "concept:name",):
    """
    Create a variant-compressed XES log and the corresponding mapping JSON.

    The compressed log contains exactly one trace per discovered variant, 
    enriched with variant metadata such as index and frequency. The JSON mapping 
    preserves the correspondence between variant indices, original case 
    identifiers, and activity sequences:
        - trace concept:name = "VAR_<idx>",
        - trace variant:idx, variant:freq.
    Saves the JSON mapping idx -> freq -> case_ids -> variant_key.
    """
    variants = build_variants_from_xes(original_xes, activity_key, case_id_key)
    out_log = EventLog()
    mapping = {"variants": []}
    for v in variants:
        tr_new = Trace()
        tr_new.attributes["concept:name"] = f"VAR_{v['variant_idx']}"
        tr_new.attributes["variant:idx"] = int(v["variant_idx"])
        tr_new.attributes["variant:freq"] = int(v["freq"])
        for ev in v["example_trace"]:
            e = Event()
            for k, val in ev.items():
                e[k] = val
            if activity_key not in e:
                e[activity_key] = ev.get(activity_key, "")
            tr_new.append(e)
        out_log.append(tr_new)
        mapping["variants"].append({
            "variant_idx": int(v["variant_idx"]),
            "freq": int(v["freq"]),
            "case_ids": v["case_ids"],
            "variant_key": list(v["variant_key"])})
    out_xes.parent.mkdir(parents=True, exist_ok=True)
    xes_exporter.apply(out_log, str(out_xes))
    out_mapping_json.parent.mkdir(parents=True, exist_ok=True)
    out_mapping_json.write_text(json.dumps(mapping,
                                           indent=2,
                                           ensure_ascii=False),
                                           encoding="utf-8")

    return mapping


def expand_cortado_variant_csv_to_cases(cortado_csv_no_header: Path,
                                        mapping_json: Path,
                                        out_csv: Path,) -> pd.DataFrame:
    """
    Expands Cortado's output (which has 1 row per variant, with idx=variant_idx)
    to 1 row per original case. Requires that in the Cortado's CSV the idx
    column is the same idx of the traces in the compressed log (VAR_<idx>).
    """
    df = parse_cortado_unfolding_csv_no_header(cortado_csv_no_header)
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    idx2meta = {int(v["variant_idx"]): v for v in mapping["variants"]}
    rows = []
    for _, r in df.iterrows():
        vidx = int(r["idx"])
        meta = idx2meta.get(vidx)
        if not meta:
            continue
        for cid in meta["case_ids"]:
            rows.append({
                "case_id": cid,
                "variant_idx": vidx,
                "variant_freq": int(meta["freq"]),
                "variant_key": " -> ".join(meta["variant_key"]),
                "cortado_cost": (float(r["cost"])
                                 if pd.notna(r["cost"])
                                 else None),
                "trace_len": (int(r["trace_len"])
                              if pd.notna(r["trace_len"])
                              else None),
                "time_1": (float(r["time_1"])
                           if pd.notna(r["time_1"])
                           else None),
                "time_2": (float(r["time_2"])
                           if pd.notna(r["time_2"])
                           else None),
                "size_1": (int(r["size_1"])
                           if pd.notna(r["size_1"])
                           else None),
                "size_2": (int(r["size_2"])
                           if pd.notna(r["size_2"])
                           else None),
            })
    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    return df_out


def weighted_conformance_from_cortado_variants(cortado_csv_no_header: Path,
                                               mapping_json: Path
                                               ) -> Dict[str, float]:
    """
    Compute strict conformance from variant-level Cortado results using case 
    weights.

    The function combines the raw Cortado costs with the original frequency of 
    each variant to recover a case-weighted strict conformance percentage over 
    the full sampled log.

    conformant% = (sum(freq of variants with cost==0) /
                    sum(freq of all variants)) * 100
    """
    df = parse_cortado_unfolding_csv_no_header(cortado_csv_no_header)
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    idx2freq = {int(v["variant_idx"]): int(v["freq"]) 
                for v 
                in mapping["variants"]}
    total = 0
    ok = 0
    for _, r in df.iterrows():
        vidx = int(r["idx"])
        freq = idx2freq.get(vidx, 0)
        total += freq
        if (pd.notna(r["cost"]) and float(r["cost"]) == 0.0):
            ok += freq
    pct = (ok / total * 100.0) if total else 0.0
    return {"total_cases": float(total),
            "conformant_cases": float(ok),
            "conformant_pct_weighted": float(pct)}


def get_case_ids_in_log_order(xes_path: Path,
                              case_id_key: str = "concept:name") -> List[str]:
    """
    Return case identifiers in the same order as traces appear in a XES log.

    This ordering is required to align PM4Py per-trace results with the original
    case identifiers when building one-to-one comparison reports.
    """
    log = xes_importer.apply(str(xes_path))
    out = []
    for tr in log:
        cid = tr.attributes.get(case_id_key)
        out.append(str(cid) if cid is not None else "")

    return out


def build_comparison_report_per_case(sampled_xes_path: Path,
                                     pm4py_details_json: Path,
                                     cortado_expanded_per_case_csv: Path,
                                     out_csv_path: Path,
                                     case_id_key: str = "concept:name"
                                     ) -> pd.DataFrame:
    """
    Build a one-to-one per-case comparison report for PM4Py and Cortado.

    The function combines the sampled XES order, the PM4Py alignment details, 
    and a Cortado per-case expanded CSV into a single table indexed by case 
    identifier. The resulting report is meant for precise case-level analysis 
    of costs, fitness, and strict conformance differences.
    """
    details = json.loads(pm4py_details_json.read_text(encoding="utf-8"))
    df_pm = []
    for i, a in enumerate(details, start=1):
        cost = a.get("cost",
                     a.get("alignment_cost",
                           a.get("alignment_costs", None)))
        bwc = a.get("bwc", None)
        fit_printed = a.get("fitness", None)
        fit_exact = None
        if cost is not None and bwc is not None:
            try:
                bwc_f = float(bwc)
                if bwc_f > 0:
                    fit_exact = 1.0 - (float(cost) / bwc_f)
            except Exception:
                pass
        df_pm.append({
            "idx": i,
            "pm4py_cost": (float(cost)
                           if cost is not None
                           else None),
            "pm4py_bwc": (float(bwc)
                          if bwc is not None
                          else None),
            "pm4py_fitness_printed": (float(fit_printed)
                                      if fit_printed is not None
                                      else None),
            "pm4py_fitness_exact": (float(fit_exact)
                                    if fit_exact is not None
                                    else None),
            "pm4py_conformant_strict": ((float(cost) == 0.0)
                                        if cost is not None
                                        else None),
        })
    df_pm = pd.DataFrame(df_pm)
    case_ids = get_case_ids_in_log_order(sampled_xes_path,
                                         case_id_key=case_id_key)
    if len(case_ids) != len(df_pm):
        raise ValueError(f"Mismatch: case_ids nel log={len(case_ids)} vs " +
                         f"PM4Py alignments={len(df_pm)}")
    df_pm["case_id"] = case_ids
    df_c = pd.read_csv(cortado_expanded_per_case_csv)
    if ("case_id" not in df_c.columns) or ("cortado_cost" not in df_c.columns):
        raise ValueError(f"CSV Cortado espanso deve avere almeno: " +
                         f"case_id, cortado_cost")
    df_pm["case_id"] = df_pm["case_id"].astype(str)
    df_c["case_id"] = df_c["case_id"].astype(str)
    df_out = df_pm.merge(df_c,
                         on="case_id",
                         how="left",
                         suffixes=("", "_cortado"))
    df_out["pm4py_fitness_exact_10"] = (
        df_out["pm4py_fitness_exact"].apply(fmt10))
    ensure_dir(out_csv_path.parent)
    df_out.to_csv(out_csv_path, index=False)

    return df_out