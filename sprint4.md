Excellent. With Sprints 1-3, we have successfully built and validated the core components of the framework: a sophisticated conditional volumetric model, a high-precision point-cloud model, and a robust uncertainty quantification pipeline. The final sprint, Sprint 4, is about synthesis, discovery, and delivery. It ties everything together, performs the novel scientific analyses, and packages the entire project for public release with critical interoperability features to ensure broad scientific accessibility.

Here is the plan for Sprint 4.

---

### **Sprint 4: Synthesis, Discovery, and Interoperable Release**

**Duration:** 3 Weeks
**Core Goal:** To leverage the powerful models to generate novel scientific insights, perform the final analyses (Hierarchy, Trajectory), and—critically—package all project artifacts in open, language-agnostic formats for broad public use and long-term preservation.

**Key Demo KPI (The "Payload"):** **The "Complete & Open Discovery Platform."** The final, public-ready dashboard integrates all previous demos and adds two new, powerful views:
1.  **The "Hierarchy Explorer":** A tree diagram view of the learned cognitive hierarchy from the H-VAE prototype. Clicking nodes reveals canonical brain maps for different cognitive levels.
2.  **The "Neuroscience Zeitgeist" Timeline:** An animated timeline from 1995-2024 with UMAP plots and representative brain maps showing research evolution.

The dashboard includes a "Downloads" section where users can directly access interoperable artifacts (CSV, ONNX, NIfTI files), demonstrating how latent vectors and ONNX models work together across platforms.

**Criteria for Sprint Completion & Project Conclusion:**
1.  All planned scientific analyses (Cross-model CCA, H-VAE, Trajectory GRU) are complete and logged.
2.  The two primary manuscripts are drafted with all key figures and tables.
3.  **All key data and model artifacts are successfully exported to non-Python-specific formats (ONNX, CSV, Parquet, NIfTI).**
4.  The final public dashboard is deployed, functional, and links to downloadable open-format artifacts.
5.  All repositories are documented with clear instructions for both Python and language-agnostic usage.

---

#### **Granular Tickets for Sprint 4**

**Epic 1: Cross-Model Synthesis & Hierarchical Analysis**

*   **Ticket S4.1.1: `analysis:run_cross_model_cca`**
    *   **Description:** Encode the entire test set with both the final volumetric C-VAE and the point-cloud C-VAE. Perform Canonical Correlation Analysis (CCA) on the two sets of latent vectors to find the shared and unique dimensions of variance between the two models.
    *   **Acceptance Criteria:** A report and a correlation matrix plot are generated, identifying the top 3 shared canonical components.

*   **Ticket S4.1.2: `analysis:train_h-vae_prototype`**
    *   **Description:** Execute the pre-defined H-VAE prototype plan. Implement the two-level hierarchical VAE with protected skip paths and per-level KL annealing. Train it on the 25% data subset.
    *   **Acceptance Criteria:** The training run completes. A validation script checks the model against the stage-gate criteria (reconstruction MSE, modularity, ontology alignment). A "go/no-go" decision is logged.

*   **Ticket S4.1.3: `analysis:run_hdbscan_clustering`**
    *   **Description:** Concatenate the latent vectors from the two primary models (or use the H-VAE's if it passed the gate). Run HDBSCAN to perform density-based clustering on the studies in latent space.
    *   **Acceptance Criteria:** Cluster labels are generated for all test studies. A script automatically profiles the top 5 largest clusters by their most frequent cognitive terms, revealing emergent study groupings.

**Epic 2: The Latent Trajectory & "Zeitgeist" Model**

*   **Ticket S4.2.1: `analysis:create_latent_timeseries`**
    *   **Description:** Using the final volumetric C-VAE, encode every study in the entire dataset. Order the resulting latent vectors (`z`) chronologically by publication date to create a high-dimensional time series.
    *   **Acceptance Criteria:** A single `.pt` or `.npy` file containing the time-ordered latent vectors is saved.

*   **Ticket S4.2.2: `analysis:train_trajectory_gru`**
    *   **Description:** Implement and train a simple Gated Recurrent Unit (GRU) model to predict `z_t+1` from `z_t`. Use the hold-one-year-out evaluation strategy (e.g., train on 1995-2018, validate on 2019).
    *   **Acceptance Criteria:** The GRU model is trained and its Fréchet Distance forecast quality is computed and logged.

*   **Ticket S4.2.3: `demo:build_zeitgeist_timeline`**
    *   **Description:** Create the "Neuroscience Zeitgeist" animated dashboard view. This involves pre-computing the UMAP embedding and the mean decoded map for each year. The UI will be a simple slider that updates the plot and the brain viewer.
    *   **Acceptance Criteria:** The animated timeline demo is functional and provides a compelling narrative of how neuroimaging research has evolved.

**Epic 3: Interoperable Export & Release**

*   **Ticket S4.3.1: `release:export_tabular_data`**
    *   **Description:** Export all tabular data products—including the final latent vectors, study metadata, cluster labels, and trajectory coordinates—to two formats: **CSV** (for universal accessibility) and **Apache Parquet** (for high-performance, type-safe access in data science environments like R and Python).
    *   **Acceptance Criteria:** A `data/` directory in the release contains `.csv` and `.parquet` files. A schema description file (`schema.json`) defines the columns and data types.

*   **Ticket S4.3.2: `release:export_models_to_onnx`**
    *   **Description:** Convert the final, trained model architectures to the **ONNX (Open Neural Network Exchange)** format. This will create separate, portable files for the key components:
        1.  Volumetric C-VAE Encoder
        2.  Volumetric C-VAE Decoder
        3.  Point-Cloud C-VAE Encoder
        4.  Point-Cloud C-VAE Decoder
    *   **Acceptance Criteria:** Four `.onnx` files are generated. A validation script successfully loads each ONNX model using `onnxruntime` and performs a forward pass, yielding numerically close results to the original PyTorch model.

*   **Ticket S4.3.3: `docs:create_non_python_usage_example`**
    *   **Description:** Create a small, self-contained example demonstrating how to use the ONNX models and CSV files in another language. An R script using the `onnx` and `reticulate` packages, or a JavaScript snippet using `onnxruntime-web`, would be ideal. This serves as a "Rosetta Stone" for non-Python users.
    *   **Acceptance Criteria:** The example script is included in the repository and successfully loads a latent vector from the CSV and decodes it using the ONNX decoder model.

*   **Ticket S4.3.4: `release:standardize_brain_map_outputs`**
    *   **Description:** Ensure all generated brain volumes (e.g., from latent traversals, cluster means) are saved in the standard **NIfTI (`.nii.gz`)** format, which is the universal standard for neuroimaging and readable by virtually all relevant software (SPM, FSL, AFNI, R, MATLAB).
    *   **Acceptance Criteria:** All final image outputs are in NIfTI format.

**Epic 4: Final Documentation & Dissemination** (Revised)

*   **Ticket S4.4.1: `docs:write_manuscripts_and_figures`**
    *   *(No changes)*

*   **Ticket S4.4.2: `docs:update_documentation_for_interoperability`**
    *   **Description:** Significantly update all `README.md` files to include a dedicated section on "Using Our Results Outside of Python." This section will explain the open formats, detail the data schemas, and link directly to the non-Python usage example.
    *   **Acceptance Criteria:** The documentation explicitly guides users on how to leverage the CSV, Parquet, and ONNX files.

*   **Ticket S4.4.3: `release:publish_all_artifacts`**
    *   **Description:** Upload all final artifacts to public repositories. This now includes the `.onnx`, `.csv`, and `.parquet` files alongside the Python-specific checkpoints and code.
    *   **Acceptance Criteria:** All artifacts are public on GitHub/Zenodo/Hugging Face and are clearly organized.

*   **Ticket S4.4.4: `release:deploy_final_dashboard`**
    *   **Description:** Deploy the final web application. Add a "Download Artifacts" section to the dashboard UI that links directly to the key public files (e.g., "Download Latent Space as CSV," "Download Decoder Model as ONNX").
    *   **Acceptance Criteria:** The public dashboard is live and provides clear, direct access to the interoperable data and model files.

---

### **Summary of Language-Agnostic Artifacts**

| Artifact Type | Primary Open Format | Rationale |
| :--- | :--- | :--- |
| **Tabular Data** (Latent Vectors, Metadata) | **CSV & Apache Parquet** | CSV for universal, simple access. Parquet for efficient, type-safe access in modern data science tools. |
| **Neural Network Models** (Encoders/Decoders) | **ONNX** | The industry standard for model interoperability, runnable in C++, Java, JS, R, etc. |
| **Volumetric Brain Maps** | **NIfTI (.nii.gz)** | The universal standard for fMRI data, compatible with all neuroimaging software. |
| **Point-Cloud Coordinates** | **CSV** | A simple, text-based format easily parsed by any language. |