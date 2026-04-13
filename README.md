# Process Mining Thesis Project – Alignments and Unfoldings

## 📌 Overview

This repository contains the implementation developed for a Master's thesis focused on **process mining**, with particular emphasis on:

- Conformance checking via **alignments**
- Optimization using **unfoldings**
- Experimental comparison between standard approaches and unfolding-based techniques

The project builds upon and extends the functionality of the PM4Py library and a modified version of the ariva-work/cortado-core repository, introducing custom modifications to support experimental evaluation and performance analysis.

---

## 🎯 Objectives

The main goals of this project are:

- Implement a pipeline for conformance checking using:
  - Standard alignments (via PM4Py)
  - Unfolding-based alignments (via cortado-core)
- Compare performance in terms of:
  - Runtime
  - Memory usage
- Analyze the impact of log characteristics (e.g., concurrency, trace length)
- Investigate the behavior of unfolding techniques on complex real-world datasets

---

## ⚙️ Installation

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules https://github.com/GiorgioCasoni/ConformanceCheckingPipeline
```

## 🚀 Usage

The main entry point of the project is:
```bash
python src/main_script.py
```
Expected functionality:
1. Load event logs (XES format)
2. Perform sampling (if enabled)
3. Compute:
    - Standard alignments (PM4Py)
    - Unfolding-based alignments (cortado-core)
4. Export results:
    - JSON (PM4Py)
    - CSV (cortado-core)
    - Generate performance metrics

## 📊 Experimental Setup

The experiments are designed to:
- Work on real-life event logs
- Evaluate scalability with increasing number of traces
- Compare:
    - Exact alignments
    - Approximate unfolding-based techniques

Logs used include BPI datasets and other real-world logs with varying:
- Concurrency
- Loop complexity
- Trace length

## 🔧 cortado-core Integration

This project uses a modified version of cortado-core.

Key points:

- The library is included as a Git submodule:
    ```bash
    external/cortado-core
    ```

- The version used is a fork maintained for this thesis:

    👉 https://github.com/GiorgioCasoni/cortado-core

- The original project is available at:

    👉 https://github.com/ariba-work/cortado-core

## ✏️ Modifications to cortado-core

Several modifications have been introduced to support:
- Improved performance monitoring
- Better control over unfolding execution
- Integration with the experimental pipeline
- Custom output handling (CSV / logging)

⚠️ Important:

This repository does not contain a detailed description of the modifications.

👉 A dedicated documentation (or README) for the forked repository will provide:
- Exact list of modified files
- Description of changes
- Rationale behind modifications

## ⚖️ License

This project is licensed under the:

**GNU General Public License v3.0 (GPL-3.0)**

This is required because the project includes and modifies GPL-licensed code from cortado-core
Implications:
- The code is open-source
- Any derivative work must also be distributed under GPL-3.0

See the `LICENSE` file for details.