Excellent. Sprint 1 established the foundational data pipeline and proved the core generative concept is viable with a baseline model. Sprint 2 will build directly on this success, focusing on sophistication: introducing the advanced conditioning mechanisms, hardening the model architecture, and formalizing the evaluation metrics.

Here is the plan for Sprint 2.

---

### **Sprint 2: Advanced Conditioning & Architectural Hardening**

**Duration:** 3 Weeks
**Core Goal:** To evolve the baseline VAE into the full **Conditional β-VAE (C-β-VAE)** specified in the final proposal. This involves integrating metadata conditioning (FiLM), adversarial de-biasing, and the more robust DenseNet/dilated convolution backbone. The objective is to produce a model that not only generates brain maps but generates them *conditionally* and is verifiably less biased by publication year.

**Key Demo KPI (The "Payload"):** **The "Counterfactual Machine."** An enhanced version of the Sprint 1 demo. This interactive dashboard will have sliders for latent variables *and* dropdown menus/sliders for conditioning variables (e.g., `Task Category`, `Sample Size`, `Year`). The user can select a latent code from a "Working Memory" study, then toggle the "Year" from 1998 to 2023 and instantly see how the model predicts the activation map would change, demonstrating the power of conditional generation.

**GPU Requirements & Paperspace Integration:**
Sprint 2 introduces significantly larger models (DenseNet + FiLM + GRL) requiring GPU acceleration. The M3 MacBook used in Sprint 1 will be insufficient for the computational demands.

**Solution: Paperspace Gradient Integration**
- **Platform**: Paperspace Gradient for console-based GPU training
- **Cost**: ~$0.50-0.76/hour, estimated $15-25 total for Sprint 2
- **Setup**: One-time CLI installation and authentication
- **Workflow**: Develop locally, train on cloud GPU, sync results automatically

**Criteria for Sprint Completion & Proceeding to Sprint 3:**
1.  The C-β-VAE model trains end-to-end with both FiLM conditioning and the GRL de-biasing head.
2.  The validation loss for the C-β-VAE is lower than the baseline VAE from Sprint 1, demonstrating the benefit of conditioning.
3.  The adversarial "year predictor" head shows a validation accuracy that is only slightly better than chance (~5-10% for 10 year-bins), proving the de-biasing is effective.
4.  The "Counterfactual Machine" demo is functional and shows qualitatively plausible changes in generated maps when conditioning variables are altered.
5.  **Paperspace GPU training pipeline is functional and cost-effective.**

---

#### **Granular Tickets for Sprint 2**

**Epic 0: Paperspace GPU Setup & Integration**

*   **Ticket S2.0.1: `infra:setup_paperspace_cli`**
    *   **Description:** Install Paperspace Gradient CLI locally and authenticate. Create project for sematlas training runs. Test basic job submission with a simple PyTorch example.
    *   **Acceptance Criteria:** `gradient --version` works. Can successfully create and run a test job. Project `sematlas` exists in Paperspace account.

*   **Ticket S2.0.2: `infra:create_training_scripts`**
    *   **Description:** Create console-based scripts for Paperspace training orchestration: `scripts/paperspace_train.sh`, `scripts/monitor_training.sh`, `scripts/download_results.sh`, and `scripts/cost_report.sh`.
    *   **Acceptance Criteria:** Scripts exist and have proper permissions. Each script has clear usage documentation. Test run with dummy job completes successfully.

*   **Ticket S2.0.3: `infra:configure_environment_sync`**
    *   **Description:** Configure automatic upload of code, data, and configs to Paperspace. Ensure environment.yml and requirements are properly handled. Set up automatic download of training artifacts.
    *   **Acceptance Criteria:** Local code changes sync to Paperspace automatically. Training results download to correct local directories. Environment reproducibility verified.

*   **Ticket S2.0.4: `infra:integrate_wandb_paperspace`**
    *   **Description:** Ensure W&B logging works seamlessly from Paperspace training runs. Configure API keys and logging to be accessible from both local development and cloud training.
    *   **Acceptance Criteria:** W&B dashboard shows metrics from Paperspace training runs. Local development and cloud training logs appear in same project. API key management secure and functional.

**Epic 1: Implementing Advanced Model Architecture**

*   **Ticket S2.1.1: `model:upgrade_backbone_to_densenet`**
    *   **Description:** Replace the ResNet-10 backbone from Sprint 1 with a 3D DenseNet architecture. Implement the dilated convolutions in the final block to ensure a full receptive field.
    *   **Acceptance Criteria:** The new model instantiates correctly and can process a batch of data. A parameter count confirms it is within the expected range.

*   **Ticket S2.1.2: `model:implement_metadata_imputation`**
    *   **Description:** Add the amortization head to the model to handle missing metadata. The head should take the input image `x` and output a distribution (`μ`, `log σ²`) for `field_strength` and `fwhm`.
    *   **Acceptance Criteria:** The VAE forward pass now also returns imputed metadata values and their uncertainty. The loss function is updated to include a small imputation loss term.

*   **Ticket S2.1.3: `model:implement_film_conditioning`**
    *   **Description:** Implement the FiLM (Feature-wise Linear Modulation) generator and FiLM layers. The generator will be a small MLP that takes the complete (real + imputed) metadata vector and produces `γ` and `β` parameters. These will modulate the feature maps in both the encoder and decoder DenseNet backbones.
    *   **Acceptance Criteria:** FiLM layers are integrated into the DenseNet blocks. A forward pass with a dummy metadata vector runs successfully.

*   **Ticket S2.1.4: `model:implement_grl_debiasing`**
    *   **Description:** Implement the Gradient Reversal Layer (GRL) and the small adversarial MLP head for predicting publication year from the latent `z`. Add the adversarial loss term (binary cross-entropy) to the main VAE loss.
    *   **Acceptance Criteria:** The new adversarial loss is logged to W&B. The `λ` for the GRL can be scheduled (e.g., via a PyTorch Lightning callback).

**Epic 2: Hardening Training and Evaluation**

*   **Ticket S2.2.1: `train:update_datamodule_with_metadata`**
    *   **Description:** Modify the `LightningDataModule` from Sprint 1 to also load and serve the conditioning metadata (task category, year, sample size, etc.) alongside the image volumes.
    *   **Acceptance Criteria:** The `train_dataloader` now yields batches of `(image, metadata)`.

*   **Ticket S2.2.2: `train:implement_kl_controller`**
    *   **Description:** Write a PyTorch Lightning callback that monitors the KL divergence term. If the KL-to-total-loss ratio falls below a threshold (e.g., 0.05) for 3 consecutive epochs, it should increase the `β` hyperparameter of the model.
    *   **Acceptance Criteria:** A log in W&B shows the `β` value changing during a test run where KL is artificially suppressed.

*   **Ticket S2.2.3: `train:refine_optimizer_and_scheduler`**
    *   **Description:** Update the `configure_optimizers` method to use AdamW with `β₂=0.995`. Implement the scheduled ramp for the GRL's `λ` hyperparameter.
    *   **Acceptance Criteria:** Hyperparameters are correctly logged in W&B. A plot of `λ` over training steps shows the correct ramp.

*   **Ticket S2.2.4: `eval:implement_formal_metrics`**
    *   **Description:** Create an evaluation script (`evaluate.py`) that loads a trained model and computes the formal metrics on the test set: Voxel-wise Pearson `r`, SSIM, and the balanced accuracy of the adversarial year-predictor head.
    *   **Acceptance Criteria:** The script produces a final `test_results.json` file with all key metrics after running on a trained checkpoint.

**Epic 3: The "Counterfactual Machine" Demo**

*   **Ticket S2.3.1: `demo:upgrade_inference_wrapper`**
    *   **Description:** Modify the Sprint 1 inference wrapper. The `.decode(z, m)` method must now accept both a latent vector `z` and a metadata vector `m`.
    *   **Acceptance Criteria:** The wrapper can successfully generate a map given a latent code and a dictionary of conditioning variables.

*   **Ticket S2.3.2: `demo:build_conditional_dashboard`**
    *   **Description:** Evolve the Sprint 1 notebook into a simple Streamlit or Dash web app. Create UI elements (dropdowns for categorical metadata, sliders for continuous metadata) that construct the metadata vector `m`. The app should allow a user to fix `z` and interactively change `m` to see the effect on the decoded brain map.
    *   **Acceptance Criteria:** The dashboard is functional and deployed locally. Changing a dropdown (e.g., "Task: Emotion" to "Task: Memory") produces a visibly different and plausible brain map in the viewer.