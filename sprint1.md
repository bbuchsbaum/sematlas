Excellent. Structuring this complex project into agile sprints is the perfect way to manage development, maintain momentum, and ensure regular delivery of tangible results. Each sprint will have a clear goal, a "Key Demo" that serves as a visible payload, and granular tickets for the team to execute.

Here is the plan for Sprint 1.

---

### **Sprint 1: Data Foundation & Baseline Model Viability**

**Duration:** 3 Weeks
**Core Goal:** To establish a fully versioned, hardened data pipeline and train a baseline, non-conditional 3D VAE to prove the core concept is viable. The goal is not a perfect model, but a functional end-to-end pipeline that produces a "blurry but recognizable" brain map from a latent code.

**Key Demo KPI (The "Payload"):** **The "Latent Slider."** An interactive Jupyter notebook or a barebones Streamlit app where a user can move a single slider corresponding to one latent dimension. As the slider moves, a 3D brain map in the viewer updates in real-time, showing the first-ever generative traversal of the Neurosynth latent space. This is the "wow" moment that proves the concept works.

**Criteria for Sprint Completion & Proceeding to Sprint 2:**
1.  The data pipeline runs end-to-end without manual intervention and is versioned with DVC.
2.  The baseline VAE model trains to completion without `NaN` losses and demonstrates a validation reconstruction loss (MSE) that is clearly decreasing.
3.  The "Latent Slider" demo is functional and shows qualitatively distinct (even if noisy) brain patterns as the latent variable is changed. The decoded outputs must be anatomically plausible (i.e., within the brain mask).

---

#### **Granular Tickets for Sprint 1**

**Epic 1: Hardened Data Curation Pipeline**

*   **Ticket S1.1.1: `data:setup_neurosynth_download`**
    *   **Description:** Write and document a script using `NiMARE` to download the latest Neurosynth database (coordinates, metadata, abstracts).
    *   **Technical Reference:** See Appendix1.md A1.2 for Neurosynth specifications and A1.3.3 for NiMARE integration details.
    *   **Acceptance Criteria:** Data is downloaded to a specified `data/raw` directory. Script is runnable via a `Makefile` command.

*   **Ticket S1.1.2: `data:implement_deduplication`**
    *   **Description:** Implement the directional deduplication logic. Create a function that takes a pandas DataFrame of study contrasts and returns a deduplicated version based on the hash of (rounded coordinates + t-stat sign).
    *   **Technical Reference:** See proposal.md Section 2.2 for directional deduplication specification.
    *   **Acceptance Criteria:** Unit test passes with a known input/output pair. A log file is produced listing the number of original vs. retained contrasts.

*   **Ticket S1.1.3: `data:validate_coordinate_space`**
    *   **Description:** Validate that Neurosynth coordinates are in expected MNI152 space ranges and create validation logging. Neurosynth has already preprocessed all coordinates to MNI152 space, so validation confirms data quality without applying transformations.
    *   **Technical Reference:** See Appendix1.md A1.2.4 for coordinate space preprocessing understanding and A1.3.4 for NiMARE validation utilities.
    *   **Acceptance Criteria:** A `coordinate_validation_log.json` is created. Validation confirms coordinates are within MNI152 bounds. No coordinate transformations are applied - data is preserved exactly as-is.

*   **Ticket S1.1.4: `data:create_volumetric_cache`**
    *   **Description:** Write the script to convert the final coordinate set into NIfTI volumes using the dual-kernel (6mm/12mm) augmentation strategy. Cache the resulting 4D tensors into a high-speed LMDB database for fast training.
    *   **Technical Reference:** See Appendix1.md A1.3.3 for kernel transformers and proposal.md Section 2.1 for augmentation specifications.
    *   **Acceptance Criteria:** LMDB file is created. A separate utility can successfully read a random study ID from the cache and return a correctly shaped PyTorch tensor.

*   **Ticket S1.1.5: `data:finalize_splits_and_dvc`**
    *   **Description:** Create and save the final stratified train/validation/test splits (70/15/15). Initialize DVC for the entire data pipeline (`dvc run ...`), versioning the final cached data.
    *   **Acceptance Criteria:** `dvc.yaml` and `.dvc` files are committed. A team member can run `dvc pull` and `dvc repro` successfully.

**Epic 2: Baseline 3D VAE Implementation**

*   **Ticket S1.2.1: `model:define_resnet_vae_architecture`**
    *   **Description:** Implement the 3D ResNet-10 VAE architecture in PyTorch. Use Group Normalization and ensure the encoder outputs `μ` and `log σ²`. The decoder should take a latent vector `z` and produce a full 3D volume.
    *   **Acceptance Criteria:** Model can be instantiated. A dummy tensor of the correct input shape can be passed through the model without error.

*   **Ticket S1.2.2: `model:create_pytorch_lightning_datamodule`**
    *   **Description:** Create a `pl.LightningDataModule` to handle loading data from the LMDB cache, applying the random kernel selection, and serving batches for train/val/test.
    *   **Acceptance Criteria:** `datamodule.setup()` runs correctly. `datamodule.train_dataloader()` returns a batch of the correct shape and type.

*   **Ticket S1.2.3: `model:create_pytorch_lightning_module`**
    *   **Description:** Create a `pl.LightningModule` for the VAE. It should implement the VAE loss function (Reconstruction + KL Divergence), the reparameterization trick, and the `training_step`, `validation_step`, and `configure_optimizers` methods.
    *   **Acceptance Criteria:** The module can be initialized with the ResNet model.

*   **Ticket S1.2.4: `train:setup_wandb_and_train_script`**
    *   **Description:** Write the main `train.py` script that initializes the model, datamodule, and a PyTorch Lightning `Trainer`. Integrate Weights & Biases logging for loss curves and hyperparameters.
    *   **Acceptance Criteria:** Running `python train.py` starts a training run that successfully logs metrics to W&B for at least one epoch.

**Epic 3: The "Latent Slider" Demo**

*   **Ticket S1.3.1: `demo:create_inference_wrapper`**
    *   **Description:** Write a simple class or function that can load a trained model checkpoint from W&B and has a `.decode(z)` method that takes a latent tensor and returns a NumPy brain volume.
    *   **Acceptance Criteria:** The wrapper can successfully load the checkpoint and generate a non-zero brain map from a `torch.randn(1, 32)` vector.

*   **Ticket S1.3.2: `demo:build_interactive_notebook`**
    *   **Description:** Create a Jupyter notebook using `ipywidgets` and a 3D brain viewer like `nilearn.plotting.view_img`. The notebook should have a slider for one latent dimension. On slider change, it should call the inference wrapper and update the brain view.
    *   **Acceptance Criteria:** The notebook is functional. Moving the slider from -3 to +3 results in visibly different patterns in the `nilearn` viewer. The notebook is checked into the repository.

---

## **Data Strategy Addendum for Sprint 1**

### **Development Phase (Weeks 1-2)**
All development and debugging for tickets S1.1.1 through S1.2.4 will be performed using **mock data** transitioning to **`neurosynth_subset_1k`** data cache. The CI/CD pipeline will be configured to run tests exclusively against subset data to ensure rapid feedback cycles (<5 minutes per run). The primary goal is to achieve functional correctness of the data pipeline and the baseline VAE architecture on a manageable scale.

**Key Data Requirements:**
- **S1.1.1-S1.1.3**: Transition from mock data (100 studies) to real Neurosynth subset (1,000 studies)
- **S1.1.4-S1.1.5**: Create volumetric cache and splits using `neurosynth_subset_1k`
- **S1.2.1-S1.2.4**: Develop and test VAE architecture on subset data for rapid iteration

### **Production Phase (Week 3)**
During the final 2-3 days of the sprint, we will execute the **first full-scale training run** using the complete **`neurosynth_full_12k`** dataset. This run serves dual purposes:
1. **Scale Validation**: Stress-test the entire data pipeline at production scale
2. **Baseline Model**: Generate the definitive, high-quality model checkpoint required for the "Latent Slider" demo

**Production Requirements:**
- **Full-scale training**: Complete baseline VAE training on `neurosynth_full_12k`
- **Demo validation**: "Latent Slider" must function with production-trained model
- **Performance baseline**: Establish official metrics for Sprint 2 comparison

### **Data Transition Protocol**
1. **Mock → Subset**: Replace synthetic data with real 1k subset by end of Week 1
2. **Pipeline validation**: Verify all components work with real coordinate data
3. **Subset → Full**: Execute production training run in Week 3
4. **Checkpoint validation**: Ensure demo works with both development and production models

This strategy ensures rapid development cycles while providing the scale validation necessary for production-quality Sprint 1 completion.