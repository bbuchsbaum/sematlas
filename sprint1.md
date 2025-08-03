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

*   **Ticket S1.1.3: `data:implement_talairach_correction`**
    *   **Description:** Implement the `tal2icbm` (Lancaster 2007) transformation for all coordinates. Add a checksum and logging mechanism for studies with >4mm RMSD.
    *   **Technical Reference:** See Appendix1.md A1.2.4 for coordinate space challenges and A1.3.4 for NiMARE coordinate transformers.
    *   **Acceptance Criteria:** A `mismatch_log.json` is created. Unit test verifies that known MNI coordinates are unchanged and known Talairach coordinates are transformed correctly.

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