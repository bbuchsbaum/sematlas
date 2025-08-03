Excellent. Sprint 2 successfully delivered a sophisticated, conditional generative model. The core architecture is now in place. Sprint 3 shifts focus from the primary model to its two crucial complements: the high-precision **Point-Cloud VAE** and the robust **Uncertainty Quantification** framework. This sprint runs these two streams in parallel.

Here is the plan for Sprint 3.

---

### **Sprint 3: Precision, Point-Clouds, and Principled Uncertainty**

**Duration:** 3 Weeks
**Core Goal:** To develop two parallel, critical enhancements to the overall framework. Stream A will build the Point-Cloud C-VAE, providing a view of the data that respects millimeter-scale geometry. Stream B will implement the Deep Ensemble uncertainty framework for the main volumetric C-VAE from Sprint 2, making its predictions trustworthy.

**Key Demo KPI (The "Payload"):** **The "Dual-View Confidence Explorer."** A consolidated dashboard with two panes.
1.  **Pane 1 (The Confidence View):** Shows the "Counterfactual Machine" from Sprint 2, but now with a new dropdown to select the visualization layer: "Mean Activation," "Epistemic Uncertainty (Model Ignorance)," or "Aleatoric Uncertainty (Data Noise)." Users can now see not just *what* the model predicts, but *how confident* it is, and why.
2.  **Pane 2 (The Geometry View):** Shows a 3D scatter plot of a real study's activation foci (as points). Below it, it shows the point-cloud VAE's reconstruction of those same foci. This demonstrates the model's ability to capture fine-grained spatial arrangements, a capability totally absent in classical meta-analysis.

**Criteria for Sprint Completion & Proceeding to Sprint 4:**
1.  **Stream A (Point-Cloud):** The Point-Cloud C-VAE trains successfully. Its validation Chamfer/EMD loss decreases, and it can reconstruct point clouds that are qualitatively similar to the inputs.
2.  **Stream B (Uncertainty):** The Deep Ensemble training pipeline is functional. The resulting ensemble produces well-calibrated uncertainty maps (ECE < 0.15 on a validation subset) and the demo dashboard can visualize all three output layers.
3.  The "Dual-View Confidence Explorer" is functional and provides a compelling demonstration of both precision geometry and principled uncertainty.

---

#### **Granular Tickets for Sprint 3**

**Epic 1: Stream A - The Point-Cloud C-VAE**

*   **Ticket S3.1.1: `data:create_pointcloud_cache`**
    *   **Description:** Write a script to export the deduplicated, Talairach-corrected coordinates into an HDF5 file. Each entry should be a variable-length list of (x,y,z) points, keyed by study ID.
    *   **Acceptance Criteria:** HDF5 file is created. A utility can read a study ID and return the correct coordinate array.

*   **Ticket S3.1.2: `model:implement_pointnet_vae`**
    *   **Description:** Implement the Point-Cloud C-VAE architecture using a `torch-points3d` or similar library. Include the PointNet++ backbone, Fourier feature positional encoding, and a simple MLP decoder that generates a fixed-size point set.
    *   **Acceptance Criteria:** The model can be instantiated and can process a batch of padded point clouds.

*   **Ticket S3.1.3: `model:implement_point_conditioning`**
    *   **Description:** Integrate the metadata conditioning vector `m` into the point-cloud model. A simple and effective method is to concatenate `m` to the global feature vector extracted by the PointNet++ encoder before it is passed to the latent variable head.
    *   **Acceptance Criteria:** The model's forward pass accepts both a point-cloud batch and a metadata batch.

*   **Ticket S3.1.4: `train:develop_pointcloud_trainer`**
    *   **Description:** Create a new PyTorch Lightning `LightningModule` for the point-cloud model. Implement the combined Chamfer + EMD loss. Create a `DataModule` that loads from the HDF5 cache and performs the fixed-size normalization (padding/dropout to N=30). Train in FP32.
    *   **Acceptance Criteria:** A training run starts successfully and logs the Chamfer/EMD loss to W&B.

*   **Ticket S3.1.5: `demo:build_pointcloud_viewer`**
    *   **Description:** Create a Python script or notebook that takes a study ID, loads the ground truth point cloud, passes it through the trained model's encoder-decoder, and visualizes both the original and reconstructed point clouds side-by-side in a 3D scatter plot (e.g., using `plotly` or `matplotlib`).
    *   **Acceptance Criteria:** The visualization for at least 5 test studies is generated and looks qualitatively reasonable. This forms Pane 2 of the demo.

**Epic 2: Stream B - Deep Ensemble Uncertainty**

*   **Ticket S3.2.1: `model:modify_decoder_for_aleatoric`**
    *   **Description:** Modify the volumetric C-VAE decoder from Sprint 2. It should now output two channels instead of one: the mean activation (`μ`) and the log variance (`log σ²`) for aleatoric uncertainty.
    *   **Acceptance Criteria:** The model's forward pass returns a tensor of shape `(B, 2, D, H, W)`.

*   **Ticket S3.2.2: `train:implement_ensemble_training`**
    *   **Description:** Adapt the main training script to support snapshot ensembling. Implement a cyclical learning rate scheduler (e.g., `CosineAnnealingWarmRestarts`). Add logic to save a model checkpoint at the end of each cycle. This creates the K=5 ensemble members in one training run.
    *   **Acceptance Criteria:** A single `python train.py --ensemble=5` command produces 5 distinct model checkpoints.

*   **Ticket S3.2.3: `eval:implement_calibration_metric`**
    *   **Description:** Write a function to compute the Expected Calibration Error (ECE). This involves binning model predictions by confidence and comparing the average confidence in each bin to the actual accuracy within that bin.
    *   **Acceptance Criteria:** The function runs on a set of predictions and produces a single ECE score. Unit test with known inputs confirms correctness.

*   **Ticket S3.2.4: `demo:create_ensemble_inference_wrapper`**
    *   **Description:** Write a new inference wrapper that loads all K=5 ensemble checkpoints. Its `.predict(z, m)` method should run the input through all 5 models, then compute and return the three separate outputs: mean activation, epistemic variance (variance across model means), and mean aleatoric variance.
    *   **Acceptance Criteria:** The wrapper returns a dictionary with three correctly shaped brain volumes: `{'mean', 'epistemic_unc', 'aleatoric_unc'}`.

*   **Ticket S3.2.5: `demo:build_confidence_explorer`**
    *   **Description:** Upgrade the Sprint 2 "Counterfactual Machine" dashboard. Add a dropdown menu to select which of the three output volumes from the ensemble wrapper is displayed in the nilearn viewer.
    *   **Acceptance Criteria:** The user can seamlessly switch between viewing the model's prediction, its "ignorance" map, and its "data noise" map. This forms Pane 1 of the demo.