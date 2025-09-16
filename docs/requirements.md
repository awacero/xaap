# XAAP Requirements Specification

## 1. Purpose
This document captures the functional and non-functional requirements that govern the XAAP (eXtended Automatic Analysis Pipeline) project. It formalizes the behaviors already implemented in the codebase and establishes expectations for operating, extending, and maintaining the system.

## 2. System Overview
XAAP automates the retrieval, preprocessing, detection, classification, and review of volcano-related seismic events. Configuration metadata is consolidated through the `xaapConfig` object so both the GUI (`xaap_gui.py`) and CLI (`xaap_cli.py`) can orchestrate a repeatable pipeline that draws waveform data from mSEED services, applies configurable preprocessing, executes one or more detection strategies (classic STA/LTA or deep-learning models from SeisBench), extracts features, and stores classifier outputs for analyst review.

## 3. Functional Requirements
### 3.1 Configuration Management
1. The system shall load all runtime parameters (mSEED endpoints, station inventories, processing windows, preprocessing flags, detection thresholds, deep-learning settings, output folders, and classification models) into an `xaapConfig` instance created either from the GUI parameter tree or a CLI configuration file.
2. The configuration layer shall validate access to referenced configuration files (e.g., server definitions, volcano catalogs, station metadata) during object construction, raising descriptive exceptions when files are missing or unreadable.
3. Operators shall be able to export the current GUI parameter tree to JSON for reuse, excluding transient date overrides, via the “Save State” workflow.

### 3.2 Data Acquisition
1. XAAP shall assemble the list of stations associated with the selected volcano and request waveform streams for each configured network/location/channel tuple over the specified time window, merging returned traces into a single `Stream` object per station.
2. Connection errors to the mSEED service must be surfaced to the caller so the user can adjust credentials or server selection.

### 3.3 Preprocessing
1. The preprocessing stage shall apply optional detrending, trace merging, and frequency filtering according to the active flags and filter parameters stored in `xaapConfig`.
2. Any failures encountered while applying these operations shall be logged with contextual information and propagated to the orchestrating layer for user feedback.

### 3.4 Event Detection
#### 3.4.1 STA/LTA Workflow
1. When the STA/LTA detection method is selected, the system shall execute Obspy’s `coincidence_trigger` with the configured STA, LTA, trigger-on/off thresholds, and coincidence value to produce candidate events.
2. Each detection run shall write a timestamped CSV containing the raw coincidence trigger output to the configured detection folder for traceability.

#### 3.4.2 Deep-Learning Workflow
1. XAAP shall instantiate the named SeisBench model version at runtime and use it to classify each station’s stream, converting returned picks and detections into `DetectionXaap` windows with configurable padding for subsequent processing.
2. The deep-learning pipeline shall aggregate individual detections using coincidence logic so multi-station events are retained only when the configured coincidence sum is satisfied, exporting both raw and coincidence CSV reports when detections are present.
3. GUI and CLI workflows must surface errors raised during model creation or inference so operators can change model selections or fall back to STA/LTA processing.

### 3.5 Feature Extraction and Classification
1. When classification is requested, XAAP shall build feature vectors (currently spectral and cepstral domains) for each detected window using the `aaa_features` library, scale them with `StandardScaler`, and apply the configured scikit-learn model to assign volcano-event categories (e.g., LP vs VT).
2. Classification results shall be persisted as CSV files under the configured data directory to support downstream analyst review and auditing.

### 3.6 User Interfaces
#### 3.6.1 Graphical User Interface
1. The GUI shall expose parameter editing, waveform retrieval, preprocessing, detection (STA/LTA and deep-learning), trigger plotting, and classification actions through dedicated buttons whose signals are bound to the corresponding pipeline functions and log updates.
2. The GUI shall refuse to run downstream steps (e.g., detection) when required upstream artifacts (such as the waveform stream) are absent, emitting descriptive errors for the operator.

#### 3.6.2 Command-Line Interface
1. The CLI shall accept arguments for the configuration file, detection method (`sta_lta` or `deep`), optional classification, and comparison modes, executing the same pipeline stages as the GUI headlessly.
2. CLI runs shall persist detection and classification outputs in the configured folders, mirroring the GUI behavior.

#### 3.6.3 Manual Review Tool
1. XAAP shall provide a manual check interface capable of loading classification CSV files, populating a review table, connecting to the mSEED server for waveform retrieval, and plotting individual triggers so analysts can validate or correct automated labels.

### 3.7 Output and Data Management
1. Detection, coincidence, and classification outputs must be stored under operator-configurable folders defined in the configuration to keep raw waveforms separate from derived products.
2. Output file names shall include timestamps (and model metadata when applicable) so multiple processing runs remain distinguishable.

### 3.8 Logging and Monitoring
1. Logging shall be configured via the shared INI file so CLI, GUI, and auxiliary tools emit consistent diagnostics and progress updates.
2. Pipeline stages must log success and failure states to guide operators during long-running tasks (data download, detection, classification).

### 3.9 Error Handling
1. Exceptions raised during data retrieval, preprocessing, detection, or classification shall be caught, logged with stack information where practical, and re-raised as user-friendly errors for the caller UI to display.

## 4. Non-Functional Requirements
### 4.1 Performance and Responsiveness
1. GUI-triggered operations (data request, detection, classification) should provide immediate log feedback and avoid blocking UI rendering longer than necessary, leveraging the sequential safeguards already present (e.g., checks for missing data) to prevent wasted processing cycles.
2. Coincidence detection thresholds shall remain configurable so operators can tune throughput versus false-positive trade-offs without code changes.

### 4.2 Reliability and Fault Tolerance
1. The system shall fail fast when upstream dependencies (mSEED server, configuration files, pretrained models) are unavailable, emitting actionable log messages so the operator can retry or adjust settings.
2. Manual review tooling shall remain available to reconcile automated classifications, ensuring analysts can override or confirm results before dissemination.

### 4.3 Usability
1. GUI interactions shall update a log panel with formatted messages so users receive immediate status feedback for each action.
2. Default CLI arguments shall permit headless runs using repository-provided example configuration files, supporting scripted executions for batch processing.

### 4.4 Maintainability and Extensibility
1. Detection, preprocessing, classification, and deep-learning logic shall remain encapsulated within dedicated modules under `xaap/process` so new algorithms can be swapped in with minimal impact on the orchestrating layers.
2. Configuration parsing utilities shall centralize file handling and parameter normalization, enabling additional configuration surfaces (e.g., REST endpoints) to reuse the same core object.

### 4.5 Environment and Dependencies
1. XAAP shall run on Python 3.10 within a Conda environment and depend on the libraries enumerated in the README (PyTorch/SeisBench, PyQt/PyQtGraph, scikit-learn, `aaa_features`, `get_mseed_data`, etc.).
2. Separate installation tracks shall exist for GPU-enabled and CPU-only deployments, ensuring environments without CUDA can still execute the pipeline using the CPU PyTorch wheels.

### 4.6 Hardware and Deployment Constraints
1. GPU acceleration is optional but recommended for deep-learning detection; when unavailable, the CPU-only installation path must remain functional.
2. Operators shall ensure persistent storage for output directories defined in the configuration so timestamped CSV artifacts are not lost between sessions.

### 4.7 Data Management and Security
1. Configuration files containing server credentials or station metadata shall reside in the `config` directory and be referenced by path rather than embedded directly in code, enabling standard filesystem permissions to protect sensitive endpoints.
2. Manual review exports and classifier outputs should be version-controlled outside the repository or stored in managed storage according to institutional policies, as XAAP itself writes plain CSV artifacts without additional access controls.

### 4.8 Logging and Monitoring
1. All executables (GUI, CLI, manual review) shall initialize logging through the shared configuration to keep audit trails consistent across entry points.
2. Log files should be retained alongside detection outputs to support post-mortem analysis of processing runs; operators can adjust log destinations via the shared INI file as needed.

## 5. Constraints and Assumptions
1. XAAP assumes continuous access to mSEED waveform services defined in the configuration; offline or intermittent connectivity requires reruns once connectivity is restored.
2. SeisBench model names and versions supplied in the configuration must correspond to pretrained artifacts published by SeisBench; custom models require packaging compatible with `sbm.<Model>.from_pretrained` semantics.
3. Feature definitions used by the classifier must remain synchronized with the deployed scikit-learn model to avoid schema mismatches; updating features or models should include regenerating the configuration’s feature file reference.

## 6. Future Considerations
1. Introduce automated validation routines that compare STA/LTA and deep-learning detections against labeled datasets using the existing `validate_detection` utilities to quantify performance regressions over time.
2. Expand the documentation set with architecture diagrams referenced in the GUI header comments to guide onboarding developers and stakeholders.

## 7. Feature Roadmap
### 7.1 Near-Term Enhancements (0–3 months)
1. **Consolidated Configuration Wizard** – Add a guided setup flow in the GUI that validates credentials, station metadata, and model availability before enabling pipeline execution to reduce runtime failures caused by misconfiguration.
2. **Standardized Logging Templates** – Provide reusable log formatters and rotate-on-size policies so long acquisition runs do not exhaust disk space and operators get consistent diagnostics across CLI, GUI, and background jobs.
3. **CLI Batch Runner** – Wrap existing CLI entry points in a scheduler-friendly interface (YAML manifest or job queue) to automate repeated runs across volcanoes or time windows without manual scripting.

### 7.2 Mid-Term Initiatives (3–6 months)
1. **Automated Regression Suite** – Build reproducible test datasets and notebooks that execute end-to-end detection/classification runs, capturing metrics for STA/LTA versus deep-learning pipelines so new releases can be validated prior to deployment.
2. **Model Management Workflow** – Integrate version tracking for SeisBench and scikit-learn models (hashing, metadata registry, download caching) to streamline updates and support rollbacks when models underperform.
3. **Enhanced Manual Review UX** – Extend the manual check tool with richer waveform visualization (zoom, spectral views) and annotation export to make analyst validation faster and more precise.

### 7.3 Long-Term Vision (6–12 months)
1. **Real-Time Streaming Mode** – Introduce streaming ingestion from continuous waveform feeds with incremental detection/classification so XAAP can operate as a monitoring dashboard rather than batch processor only.
2. **Plugin Architecture for Algorithms** – Refactor detection and feature extraction modules into pluggable interfaces (entry points) that allow third parties to contribute algorithms without modifying core code.
3. **Web-Based Operations Portal** – Develop a lightweight web frontend that mirrors the GUI workflows, enabling remote analysts to launch runs, review detections, and monitor system health from a browser.
