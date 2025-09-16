# XAAP System Design Document

## 1. Purpose and Scope
This document describes how the eXtended Automatic Analysis Pipeline (XAAP) is structured and the
technical decisions that guide its construction. It is intended for engineers who will maintain or
extend the codebase and assumes familiarity with the requirements and analysis artefacts.

## 2. Design Drivers
- **Primary Use Case** – Batch and interactive processing of volcano-related seismic waveforms to
detect, classify, and review events.
- **Operational Context** – Runs on operator workstations or lab servers with intermittent GPU
availability and access to institutional mSEED services.
- **Maintainability** – Encourage modular additions of new detection models, preprocessing stages,
and review workflows without rewriting the orchestration layer.
- **Observability** – Provide consistent logging, reproducible outputs, and traceable configuration
state across GUI and CLI executions.

## 3. Technology Stack
| Layer | Technology | Purpose |
| --- | --- | --- |
| Language & Runtime | Python 3.10 (Conda env) | Unified runtime for GUI, CLI, and processing code |
| UI | PyQt5, PyQtGraph | Desktop GUI for operators with waveform visualization |
| CLI | `argparse` | Headless orchestration of pipeline runs |
| Seismic Data Handling | ObsPy | Stream acquisition, manipulation, STA/LTA detection |
| Machine Learning | PyTorch, SeisBench | Deep-learning detection models and pretrained weights |
| Classical Features & ML | `aaa_features`, scikit-learn | Feature extraction and SVM-based classification |
| Data Management | Pandas, CSV, filesystem | Persistence of detection/coincidence/classification outputs |
| Logging & Config | Python `logging`, INI/JSON/CFG files | Shared diagnostics and runtime configuration |

GPU acceleration is optional; CPU-only environments fall back to PyTorch CPU builds with identical
interfaces. External dependencies include institutional mSEED servers and locally provisioned
trained models referenced in configuration files.

## 4. Architectural Overview
### 4.1 Architectural Style
XAAP is implemented as a **modular monolith**. A thin orchestration layer (GUI and CLI entry
points) coordinates domain services contained under `xaap/process`. Each service encapsulates a
pipeline stage so that functionality can evolve while keeping a single deployable application.

### 4.2 High-Level Component Layout
```
┌─────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ User Inputs │────►│ Orchestration Layer  │────►│ Processing Services   │
│ (GUI/CLI)   │      │ (xaap_gui, xaap_cli) │      │ (request, preprocess,│
└─────────────┘      └──────────────────────┘      │ detect, classify)    │
                                                   └─────────┬────────────┘
                                                             │
                        ┌────────────────────────────────────┴─────────────────────────────────┐
                        │ External Systems & Resources                                          │
                        │ • mSEED waveform services                                            │
                        │ • SeisBench pretrained models / PyTorch runtime                      │
                        │ • Configuration JSON/CFG files                                       │
                        │ • Filesystem for CSV outputs and logs                                │
                        └──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Runtime Flow
1. Operator loads or edits configuration via GUI parameter tree (`xaap_gui`) or supplies a config
   file to the CLI (`xaap_cli`).
2. Configuration utilities (`xaap/configuration/xaapConfig.py`, `xaap_configuration.py`) validate
   paths, credentials, and runtime options, exposing an `xaapConfig` object to the pipeline.
3. Acquisition services (`xaap/process/request_data.py`) fetch waveform streams per station from the
   configured mSEED endpoints.
4. Preprocessing services (`xaap/process/pre_process.py`) merge, trim, detrend, and band-pass the
   streams according to configuration flags.
5. Detection services execute either:
   - STA/LTA coincidence triggers (`xaap/process/detect_trigger.py::get_triggers`) or
   - Deep-learning classifiers with coincidence aggregation
     (`xaap/process/process_deep_learning.py`, `coincidence_trigger_deep_learning`).
6. Optional feature extraction and classification (`xaap/process/classify_detection.py`) construct
   feature vectors (`aaa_features`) and apply scikit-learn models to label events.
7. Outputs (raw triggers, coincidence summaries, classification CSVs, logs) are persisted under the
   configured directories and surfaced through the GUI log or CLI console.
8. Analysts may review results with the manual checking tool (`xaap_manual_check.py`), which reloads
   CSV outputs and retrieves waveforms for visual inspection.

### 4.4 Execution Model
Processing is single-node and sequential per pipeline run. GUI operations are triggered by user
interaction, while CLI runs execute the entire pipeline in batch mode. Long-running calls rely on
ObsPy and PyTorch to release the GIL during heavy computation, so coarse parallelism can be added by
scheduling multiple CLI runs externally.

## 5. Detailed Component Design
### 5.1 Configuration Layer (`xaap/configuration`)
- `xaapConfig`: central object storing user selections (volcano metadata, stations, detection
  thresholds, model names, output folders). Provides typed properties consumed by processing code.
- `xaap_configuration.configure_parameters_from_gui`: projects GUI parameter tree values into
  `xaapConfig` instances and persists JSON snapshots for reproducibility.
- `configure_logging`: initializes logging with the shared INI file to align GUI/CLI log formatting.

### 5.2 Acquisition & Preprocessing (`xaap/process/request_data.py`, `pre_process.py`)
- `request_data.get_stream_volcano`: constructs station lists from configuration JSON, queries the
  mSEED server, and produces ObsPy `Stream` objects per station.
- `pre_process.pre_process_stream`: handles trace sorting, merging, optional gap filling, and filter
  application. It returns cleaned streams ready for detection and writes intermediate logs when
  transformations fail.

### 5.3 Detection Engines (`xaap/process/detect_trigger.py`, `process_deep_learning.py`)
- STA/LTA detection converts configured STA/LTA windows and coincidence values into a call to
  `coincidence_trigger`, exporting per-run CSVs (`trigger_xaap_<timestamp>.csv`).
- Deep-learning detection creates SeisBench models dynamically via `getattr(sbm, model_name)` and
  executes `model.classify` or `model.annotate`. Coincidence logic mirrors the classical path so the
  downstream classifier receives a consistent `DetectionXaap` window list.
- Shared utilities convert SeisBench annotations into ObsPy picks/detections and compute summary
  statistics for logging.

### 5.4 Classification (`xaap/process/classify_detection.py`)
- Generates spectral and cepstral features with `aaa_features.FeatureVector`.
- Applies a persisted scikit-learn estimator (`pickle`) and optional `StandardScaler` to label each
  detection window.
- Outputs CSV summaries and integrates with the manual review GUI.

### 5.5 Interface Layer
- **GUI (`xaap_gui.py`)** – PyQt5 widget hierarchy with a `ParameterTree` for configuration, buttons
  mapped to pipeline actions, waveform plotting using `GraphicsLayoutWidget`, and an embedded log
  panel.
- **CLI (`xaap_cli.py`)** – Parses configuration paths and pipeline flags (`--deep`, `--sta_lta`,
  classification toggles). Runs the same services as the GUI but headlessly, writing progress to the
  console and log files.
- **Manual Review (`xaap_manual_check.py`)** – Loads detection/classification CSVs, re-queries
  waveforms for selected events, and renders plots for analyst verification.

### 5.6 External Integration
- **mSEED Servers** – Accessed through ObsPy clients; credentials and endpoints live in
  `config/*.server_configuration.json`.
- **SeisBench & PyTorch** – Models downloaded on first use; GPU detection depends on CUDA-enabled
  environments but defaults to CPU when unavailable.
- **Filesystem** – Input configuration JSON/CFG files are versioned with the repo; outputs (CSV,
  logs, saved parameter states) are timestamped under `output_*` folders defined by the user.

### 5.7 Logging and Error Handling
- Logging is centralized; each module obtains a module-level logger and inherits settings from the
  shared INI file. Errors are propagated to the caller so the GUI can present dialog messages and the
  CLI can exit with non-zero status.
- Critical failures (missing data, authentication errors, model load issues) halt downstream stages,
  keeping partial runs auditable.

## 6. Interface & API Design
### 6.1 Command-Line Interface
`xaap_cli.py` accepts the following primary arguments:
- `--config /path/to/xaap_cli_config.cfg` – required, points to a configuration profile.
- `--detection {sta_lta, deep}` – selects the detection engine (defaults can be inferred from config).
- `--classify` – toggles classification after detection.
- `--compare` – optional mode to contrast detection outputs.

Exit codes follow standard POSIX semantics (0 for success). Logs are emitted to stdout and the
configured log file.

### 6.2 GUI Actions
Key buttons (mapped via `ParameterTree`) include **Update Parameters**, **Request Data**,
**Preprocess**, **Plot Stream**, **Detection STA/LTA**, **Detection Deep Learning**, and
**Classify**. Each button validates prerequisites (e.g., data fetched before detection) and appends
status lines to the log window. Parameter states can be exported/imported as JSON snapshots.

### 6.3 Programmatic Access
Pipeline services are plain Python functions/classes and may be reused in notebooks or batch scripts.
A typical minimal usage pattern is:
```python
from xaap.configuration.xaapConfig import xaapConfig
from xaap.process import request_data, pre_process, detect_trigger

config = xaapConfig.from_json("config/EXAMPLE.xaap_gui.json")
stream = request_data.get_stream_volcano(config)
clean_stream = pre_process.pre_process_stream(config, stream)
triggers = detect_trigger.get_triggers(config, clean_stream)
```

### 6.4 HTTP/REST Endpoints
The current system does **not** expose HTTP or REST endpoints. Future web-based integrations should
wrap the orchestration layer in a FastAPI or Flask service, reusing the same processing modules.

## 7. Data & Storage Design
### 7.1 Operational Storage (Filesystem)
- **Input** – Configuration JSON, CFG, and model pickle files under `config/`.
- **Output** – Timestamped CSVs for raw triggers, coincidence aggregates, and classifier results.
  Filenames encode run timestamps (`trigger_xaap_%Y.%m.%d.%H.%M.%S.csv`).
- **Logs** – Managed via logging configuration; stored alongside outputs or in a dedicated `logs/`
  directory if configured.

CSV Schemas:
- *Trigger CSV*: columns include `time`, `station`, `coincidence_sum`, `duration`, enabling replay of
  detection results.
- *Classification CSV*: columns for detection identifiers, feature-derived attributes, classifier
  label, and confidence/probability.

### 7.2 Proposed Relational Model (Optional Extension)
For deployments requiring centralized storage, the following logical schema can replace CSV outputs:
- `processing_run(run_id, start_time, end_time, config_hash, detection_method, operator)`
- `detection_event(event_id, run_id, start_timestamp, end_timestamp, station, coincidence_sum, source)`
- `classification_result(event_id, model_name, class_label, probability, reviewer_id, reviewed_at)`
- `review_annotation(annotation_id, event_id, status, notes, waveform_uri)`

This schema preserves provenance, supports joins for analytics, and can be implemented in PostgreSQL
or SQLite depending on deployment scale.

### 7.3 Data Retention & Backup
- Operators should archive configuration snapshots with each run to guarantee reproducibility.
- Output directories can be synchronized to institutional storage (e.g., S3, on-prem NAS). Log
  rotation is recommended for long-lived monitoring deployments.

## 8. Security & Access Control
- Configuration files containing server credentials inherit filesystem permissions; sensitive data
  should not be committed to version control.
- When using shared workstations, enable OS-level controls to protect model and configuration assets.
- Future web or API layers must add authentication and role-based access to manual review endpoints.

## 9. Future Design Considerations
- Introduce a plugin registry for detection/preprocessing algorithms (entry points) to decouple core
  modules from experimental extensions.
- Evaluate asynchronous acquisition (e.g., using `asyncio` or worker queues) for streaming
  integrations.
- Standardize schema contracts for CSV/DB outputs so third-party analytics tools can consume the
  data without bespoke parsers.

---
*Document owner: XAAP engineering team. Updates should be synchronized with requirement and
analysis documents to maintain traceability.*
