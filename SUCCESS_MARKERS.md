# SUCCESS MARKERS: Strict Sprint Completion Criteria

## Critical Rules
1. **NO SPRINT ADVANCEMENT** without achieving **100% of success markers** for the current sprint
2. **ALL CRITERIA MUST BE MET** - partial success is considered failure
3. **OBJECTIVE VALIDATION REQUIRED** for all quantitative metrics
4. **FUNCTIONAL DEMOS MUST WORK** - no exceptions for "mostly working" features
5. **If any criterion fails**: Engage advanced reasoning (Gemini, O3) to analyze and resolve

---

## Sprint 1: Data Foundation & Baseline Model
**Duration**: 3 weeks  
**Failure Threshold**: ANY criterion below fails the entire sprint

### Epic 1: Data Curation Pipeline - SUCCESS CRITERIA

#### ✅ S1.1.1: Neurosynth Download
- [ ] **PASS/FAIL**: Script downloads latest Neurosynth database without errors
- [ ] **PASS/FAIL**: Data stored in `data/raw/` directory with correct file structure
- [ ] **PASS/FAIL**: `make download-neurosynth` command executes successfully
- [ ] **PASS/FAIL**: Downloaded files match expected checksums/sizes

#### ✅ S1.1.2: Directional Deduplication  
- [ ] **PASS/FAIL**: Function processes test dataset without errors
- [ ] **PASS/FAIL**: Unit test passes with known input/output pair
- [ ] **PASS/FAIL**: Log file shows >10% deduplication rate (indicating function works)
- [ ] **PASS/FAIL**: Retains contrasts with opposite t-stat signs as distinct

#### ✅ S1.1.3: Coordinate Space Validation
- [ ] **PASS/FAIL**: Neurosynth preprocessing assumptions validated (coordinates already in MNI152)
- [ ] **PASS/FAIL**: `coordinate_validation_log.json` created with coordinate bounds validation
- [ ] **PASS/FAIL**: No coordinate transformations applied (data preserved exactly as-is)
- [ ] **PASS/FAIL**: All input coordinates preserved exactly without modification

#### ✅ S1.1.4: Volumetric Cache Creation
- [ ] **PASS/FAIL**: LMDB database created successfully 
- [ ] **PASS/FAIL**: Random study retrieval returns correctly shaped PyTorch tensor
- [ ] **PASS/FAIL**: Dual-kernel (6mm/12mm) augmentation implemented
- [ ] **PASS/FAIL**: Cache size reasonable (not corrupted/oversized)

#### ✅ S1.1.5: DVC Pipeline Setup
- [ ] **PASS/FAIL**: `dvc.yaml` and `.dvc` files committed to repository
- [ ] **PASS/FAIL**: Team member can run `dvc pull` successfully  
- [ ] **PASS/FAIL**: `dvc repro` reproduces pipeline without errors
- [ ] **PASS/FAIL**: Train/validation/test splits created (70/15/15)

### Epic 2: Baseline 3D VAE - SUCCESS CRITERIA

#### ✅ S1.2.1: ResNet VAE Architecture
- [ ] **PASS/FAIL**: Model instantiates without errors
- [ ] **PASS/FAIL**: Dummy tensor passes through model successfully
- [ ] **PASS/FAIL**: Group Normalization with groups=8 implemented
- [ ] **PASS/FAIL**: Encoder outputs μ and log σ² with correct shapes

#### ✅ S1.2.2: PyTorch Lightning DataModule
- [ ] **PASS/FAIL**: `datamodule.setup()` completes without errors
- [ ] **PASS/FAIL**: `train_dataloader()` returns batch with correct shape and type
- [ ] **PASS/FAIL**: LMDB loading functions correctly
- [ ] **PASS/FAIL**: Random kernel selection (6mm/12mm) working

#### ✅ S1.2.3: Lightning Module Implementation  
- [ ] **PASS/FAIL**: VAE loss (Reconstruction + KL) implemented correctly
- [ ] **PASS/FAIL**: Reparameterization trick functional
- [ ] **PASS/FAIL**: `training_step`, `validation_step` methods work
- [ ] **PASS/FAIL**: `configure_optimizers` returns valid optimizer

#### ✅ S1.2.4: W&B Training Script
- [ ] **PASS/FAIL**: `python train.py` starts training run successfully
- [ ] **PASS/FAIL**: Metrics logged to W&B for at least 3 epochs without errors
- [ ] **PASS/FAIL**: Loss curves show expected behavior (no NaN, decreasing trend)
- [ ] **PASS/FAIL**: Validation reconstruction loss < training loss after 10 epochs

### Epic 3: Interactive Demo - SUCCESS CRITERIA

#### ✅ S1.3.1: Inference Wrapper
- [ ] **PASS/FAIL**: Wrapper loads trained checkpoint successfully
- [ ] **PASS/FAIL**: `.decode(z)` method returns non-zero brain volume
- [ ] **PASS/FAIL**: Generated brain map is anatomically plausible (within brain mask)
- [ ] **PASS/FAIL**: Different latent vectors produce visibly different outputs

#### ✅ S1.3.2: Interactive Notebook  
- [ ] **PASS/FAIL**: Jupyter notebook runs without errors
- [ ] **PASS/FAIL**: Slider interaction updates brain viewer in real-time
- [ ] **PASS/FAIL**: Moving slider from -3 to +3 shows qualitatively distinct brain patterns
- [ ] **PASS/FAIL**: `nilearn` viewer displays anatomically valid brain maps

### SPRINT 1 FINAL VALIDATION
- [ ] **CRITICAL**: End-to-end pipeline runs from raw data to functional demo
- [ ] **CRITICAL**: "Latent Slider" demo shows clear, anatomically plausible brain pattern changes
- [ ] **CRITICAL**: Training completes without NaN losses for 20+ epochs
- [ ] **CRITICAL**: All code committed and documented in repository

---

## Sprint 2: Advanced Conditioning & Architecture  
**Duration**: 3 weeks
**Builds on**: Sprint 1 SUCCESS (all criteria met)

### Epic 0: RunPod GPU Setup - SUCCESS CRITERIA

#### ✅ S2.0.1: RunPod SDK Setup
- [ ] **PASS/FAIL**: RunPod SDK installed and authenticated successfully
- [ ] **PASS/FAIL**: `runpod.get_pods()` returns successfully
- [ ] **PASS/FAIL**: Test pod creation and termination works without errors
- [ ] **PASS/FAIL**: API authentication verified with test operations

#### ✅ S2.0.2: Training Scripts Creation
- [ ] **PASS/FAIL**: All required scripts exist with executable permissions
- [ ] **PASS/FAIL**: `runpod_train.sh` successfully launches test pod
- [ ] **PASS/FAIL**: `monitor_runpod.sh` displays pod status and SSH info
- [ ] **PASS/FAIL**: `train_runpod.py` orchestrates training correctly

#### ✅ S2.0.3: Environment Synchronization
- [ ] **PASS/FAIL**: Code upload to RunPod pod functional (git/SCP)
- [ ] **PASS/FAIL**: `requirements.txt` installs correctly in container
- [ ] **PASS/FAIL**: Training artifacts download via SSH to local paths
- [ ] **PASS/FAIL**: Environment reproducibility verified between local and pod

#### ✅ S2.0.4: W&B Integration
- [ ] **PASS/FAIL**: W&B API key configuration works on RunPod
- [ ] **PASS/FAIL**: Training metrics appear in W&B dashboard from pod runs
- [ ] **PASS/FAIL**: Local and cloud training logs unified in same W&B project
- [ ] **PASS/FAIL**: No API key exposure or security issues

### Epic 1: Advanced Architecture - SUCCESS CRITERIA

#### ✅ S2.1.1: DenseNet Backbone Upgrade
- [ ] **PASS/FAIL**: 3D DenseNet architecture implemented correctly
- [ ] **PASS/FAIL**: Dilated convolutions in final block achieve >150mm receptive field
- [ ] **PASS/FAIL**: Model processes batch without memory issues
- [ ] **PASS/FAIL**: Parameter count within expected range (documented)

#### ✅ S2.1.2: Metadata Imputation  
- [ ] **PASS/FAIL**: Amortization head outputs (μ, log σ²) for missing metadata
- [ ] **PASS/FAIL**: Imputation loss term integrated into total loss
- [ ] **PASS/FAIL**: Forward pass returns imputed values with uncertainty
- [ ] **PASS/FAIL**: Uncertainty propagation via reparameterization trick works

#### ✅ S2.1.3: FiLM Conditioning
- [ ] **PASS/FAIL**: FiLM generator MLP implemented correctly  
- [ ] **PASS/FAIL**: FiLM layers integrated in both encoder and decoder
- [ ] **PASS/FAIL**: Forward pass with metadata vector completes successfully
- [ ] **PASS/FAIL**: γ and β parameters have correct shapes for feature modulation

#### ✅ S2.1.4: GRL Adversarial De-biasing
- [ ] **PASS/FAIL**: Gradient Reversal Layer implemented correctly
- [ ] **PASS/FAIL**: Adversarial MLP head (64→1 neurons) predicts publication year
- [ ] **PASS/FAIL**: Adversarial loss (BCE) logged to W&B
- [ ] **PASS/FAIL**: λ scheduling callback functional

### Epic 2: Training Hardening - SUCCESS CRITERIA

#### ✅ S2.2.1: Metadata DataModule
- [ ] **PASS/FAIL**: DataModule yields (image, metadata) batches
- [ ] **PASS/FAIL**: Metadata includes task category, year, sample size
- [ ] **PASS/FAIL**: Missing metadata handled gracefully
- [ ] **PASS/FAIL**: Batch shapes consistent across epochs

#### ✅ S2.2.2: KL Controller Implementation
- [ ] **PASS/FAIL**: Callback monitors KL-to-total-loss ratio  
- [ ] **PASS/FAIL**: β increases by 10% when KL < 90% target for 3 epochs
- [ ] **PASS/FAIL**: W&B logs show β adjustments during training
- [ ] **PASS/FAIL**: Prevents posterior collapse (KL remains >0.01)

#### ✅ S2.2.3: Optimizer Refinement
- [ ] **PASS/FAIL**: AdamW with β₂=0.995 configured correctly
- [ ] **PASS/FAIL**: GRL λ ramping schedule (epoch 20-80) implemented
- [ ] **PASS/FAIL**: Hyperparameters correctly logged in W&B
- [ ] **PASS/FAIL**: Learning rate scheduling functional

#### ✅ S2.2.4: Formal Evaluation Metrics
- [ ] **PASS/FAIL**: `evaluate.py` script runs on trained checkpoint
- [ ] **PASS/FAIL**: Voxel-wise Pearson r computed correctly
- [ ] **PASS/FAIL**: SSIM metric implemented and functional  
- [ ] **PASS/FAIL**: `test_results.json` contains all required metrics

### Epic 3: Conditional Demo - SUCCESS CRITERIA

#### ✅ S2.3.1: Conditional Inference Wrapper
- [ ] **PASS/FAIL**: `.decode(z, m)` method accepts latent + metadata
- [ ] **PASS/FAIL**: Metadata dictionary correctly formatted and processed
- [ ] **PASS/FAIL**: Generated maps show conditioning effects
- [ ] **PASS/FAIL**: Wrapper handles missing metadata gracefully

#### ✅ S2.3.2: Conditional Dashboard
- [ ] **PASS/FAIL**: Streamlit/Dash app runs locally without errors
- [ ] **PASS/FAIL**: Dropdown menus for categorical metadata functional
- [ ] **PASS/FAIL**: Sliders for continuous metadata update maps in real-time
- [ ] **PASS/FAIL**: Changing task category produces visibly different, plausible brain maps

### SPRINT 2 FINAL VALIDATION
- [ ] **CRITICAL**: C-β-VAE validation loss < Sprint 1 baseline VAE loss
- [ ] **CRITICAL**: Adversarial year predictor accuracy ≤60% (10 year-bins, indicating successful de-biasing)
- [ ] **CRITICAL**: "Counterfactual Machine" demo shows clear conditioning effects
- [ ] **CRITICAL**: FiLM conditioning produces meaningful map variations

---

## Sprint 3: Precision & Uncertainty
**Duration**: 3 weeks  
**Builds on**: Sprint 2 SUCCESS (all criteria met)

### Epic 1: Point-Cloud VAE (Stream A) - SUCCESS CRITERIA

#### ✅ S3.1.1: Point-Cloud Cache Creation
- [ ] **PASS/FAIL**: HDF5 file created with variable-length coordinate arrays
- [ ] **PASS/FAIL**: Utility successfully reads study ID and returns coordinate array
- [ ] **PASS/FAIL**: All studies from deduplicated dataset included
- [ ] **PASS/FAIL**: Coordinate format validation (x,y,z tuples)

#### ✅ S3.1.2: PointNet++ VAE Architecture
- [ ] **PASS/FAIL**: PointNet++ backbone processes padded point clouds
- [ ] **PASS/FAIL**: MLP decoder generates fixed-size point sets (N=30)
- [ ] **PASS/FAIL**: Gaussian Random Fourier Features implemented
- [ ] **PASS/FAIL**: Model instantiation and forward pass successful

#### ✅ S3.1.3: Point-Cloud Conditioning
- [ ] **PASS/FAIL**: Metadata vector concatenated to global features
- [ ] **PASS/FAIL**: Forward pass accepts (point_cloud, metadata) batches
- [ ] **PASS/FAIL**: Conditioning effects visible in generated point clouds
- [ ] **PASS/FAIL**: Architecture handles variable metadata dimensions

#### ✅ S3.1.4: Point-Cloud Training Pipeline
- [ ] **PASS/FAIL**: Combined Chamfer + EMD loss implemented
- [ ] **PASS/FAIL**: Fixed-size normalization (padding/dropout to N=30) works
- [ ] **PASS/FAIL**: FP32 training pipeline functional
- [ ] **PASS/FAIL**: W&B logging shows decreasing Chamfer/EMD loss

#### ✅ S3.1.5: Point-Cloud Visualization
- [ ] **PASS/FAIL**: 3D scatter plot visualization script functional
- [ ] **PASS/FAIL**: Original vs reconstructed point clouds displayed side-by-side
- [ ] **PASS/FAIL**: Qualitative assessment shows reasonable reconstructions for ≥5 test studies
- [ ] **PASS/FAIL**: Point clouds respect anatomical constraints

### Epic 2: Deep Ensemble Uncertainty (Stream B) - SUCCESS CRITERIA

#### ✅ S3.2.1: Aleatoric Uncertainty Decoder
- [ ] **PASS/FAIL**: Decoder outputs 2 channels: μ and log σ²
- [ ] **PASS/FAIL**: Forward pass returns tensor shape (B, 2, D, H, W)
- [ ] **PASS/FAIL**: Aleatoric loss term correctly implemented
- [ ] **PASS/FAIL**: σ² values remain positive (log σ² can be negative)

#### ✅ S3.2.2: Snapshot Ensemble Training
- [ ] **PASS/FAIL**: `python train.py --ensemble=5` produces 5 distinct checkpoints
- [ ] **PASS/FAIL**: Cyclical learning rate (CosineAnnealingWarmRestarts) implemented
- [ ] **PASS/FAIL**: Checkpoints saved at end of each cycle
- [ ] **PASS/FAIL**: Model diversity verified across ensemble members

#### ✅ S3.2.3: Calibration Metric Implementation
- [ ] **PASS/FAIL**: Expected Calibration Error (ECE) function implemented
- [ ] **PASS/FAIL**: Unit test with known inputs confirms correctness
- [ ] **PASS/FAIL**: ECE computation handles binning correctly
- [ ] **PASS/FAIL**: Function returns single scalar ECE score

#### ✅ S3.2.4: Ensemble Inference Wrapper
- [ ] **PASS/FAIL**: Wrapper loads all K=5 ensemble checkpoints
- [ ] **PASS/FAIL**: `.predict(z, m)` returns dict with 3 brain volumes
- [ ] **PASS/FAIL**: Dictionary keys: 'mean', 'epistemic_unc', 'aleatoric_unc'
- [ ] **PASS/FAIL**: All output volumes have correct shapes and value ranges

#### ✅ S3.2.5: Confidence Explorer Dashboard
- [ ] **PASS/FAIL**: Dropdown menu switches between 3 uncertainty layers
- [ ] **PASS/FAIL**: nilearn viewer displays all uncertainty types correctly
- [ ] **PASS/FAIL**: Uncertainty maps show meaningful spatial patterns
- [ ] **PASS/FAIL**: Interface responsive and functional

### SPRINT 3 FINAL VALIDATION
- [ ] **CRITICAL**: Point-Cloud VAE validation Chamfer+EMD loss decreases over training
- [ ] **CRITICAL**: Deep Ensemble ECE < 0.15 on validation subset
- [ ] **CRITICAL**: "Dual-View Confidence Explorer" fully functional with both panes
- [ ] **CRITICAL**: Point-cloud reconstructions qualitatively similar to inputs

---

## Sprint 4: Synthesis & Public Release
**Duration**: 3 weeks
**Builds on**: Sprint 3 SUCCESS (all criteria met)

### Epic 1: Cross-Model Analysis - SUCCESS CRITERIA

#### ✅ S4.1.1: Cross-Model CCA
- [ ] **PASS/FAIL**: Test set encoded by both volumetric and point-cloud VAEs
- [ ] **PASS/FAIL**: CCA analysis completes without errors
- [ ] **PASS/FAIL**: Report identifies top 3 shared canonical components
- [ ] **PASS/FAIL**: Correlation matrix plot generated and interpretable

#### ✅ S4.1.2: H-VAE Prototype
- [ ] **PASS/FAIL**: Two-level hierarchical VAE training completes
- [ ] **PASS/FAIL**: Protected skip paths prevent trivial identity mapping
- [ ] **PASS/FAIL**: Stage-gate validation criteria met (reconstruction MSE, modularity)
- [ ] **PASS/FAIL**: Go/no-go decision logged with detailed rationale

#### ✅ S4.1.3: HDBSCAN Clustering
- [ ] **PASS/FAIL**: Density-based clustering completes on latent vectors
- [ ] **PASS/FAIL**: Cluster labels generated for all test studies
- [ ] **PASS/FAIL**: Top 5 clusters profiled by cognitive terms
- [ ] **PASS/FAIL**: Emergent study groupings show interpretable patterns

### Epic 2: Temporal Analysis - SUCCESS CRITERIA

#### ✅ S4.2.1: Latent Time Series Creation
- [ ] **PASS/FAIL**: Entire dataset encoded chronologically by publication date
- [ ] **PASS/FAIL**: Time-ordered latent vectors saved (.pt/.npy format)
- [ ] **PASS/FAIL**: Temporal coverage spans expected publication years
- [ ] **PASS/FAIL**: No missing years in critical time periods

#### ✅ S4.2.2: Trajectory GRU Training
- [ ] **PASS/FAIL**: GRU model predicts z_t+1 from z_t successfully
- [ ] **PASS/FAIL**: Hold-one-year-out validation strategy implemented
- [ ] **PASS/FAIL**: Fréchet Distance forecast quality computed and logged
- [ ] **PASS/FAIL**: Model shows better-than-baseline temporal prediction

#### ✅ S4.2.3: Zeitgeist Timeline
- [ ] **PASS/FAIL**: UMAP embedding pre-computed for all years
- [ ] **PASS/FAIL**: Mean decoded maps generated for each year
- [ ] **PASS/FAIL**: Animated timeline UI functional with slider
- [ ] **PASS/FAIL**: Provides compelling narrative of research evolution

### Epic 3: Interoperable Export - SUCCESS CRITERIA

#### ✅ S4.3.1: Tabular Data Export
- [ ] **PASS/FAIL**: CSV and Parquet files created in `data/` directory
- [ ] **PASS/FAIL**: Schema description file (`schema.json`) defines all columns
- [ ] **PASS/FAIL**: Latent vectors, metadata, cluster labels all exported
- [ ] **PASS/FAIL**: Files load correctly in R and Python environments

#### ✅ S4.3.2: ONNX Model Export
- [ ] **PASS/FAIL**: 4 ONNX files generated (volumetric encoder/decoder, point-cloud encoder/decoder)
- [ ] **PASS/FAIL**: Validation script confirms numerical similarity to PyTorch models
- [ ] **PASS/FAIL**: ONNX models load successfully with `onnxruntime`
- [ ] **PASS/FAIL**: Forward pass produces expected output shapes

#### ✅ S4.3.3: Cross-Language Usage Example
- [ ] **PASS/FAIL**: Example script (R or JavaScript) included in repository
- [ ] **PASS/FAIL**: Script loads latent vector from CSV successfully
- [ ] **PASS/FAIL**: Script decodes using ONNX decoder model
- [ ] **PASS/FAIL**: Example demonstrates full cross-platform workflow

#### ✅ S4.3.4: NIfTI Standardization  
- [ ] **PASS/FAIL**: All generated brain volumes saved as NIfTI (.nii.gz)
- [ ] **PASS/FAIL**: Files conform to neuroimaging software standards
- [ ] **PASS/FAIL**: Can be loaded in SPM, FSL, AFNI, MATLAB
- [ ] **PASS/FAIL**: Proper MNI space registration confirmed

### Epic 4: Documentation & Deployment - SUCCESS CRITERIA

#### ✅ S4.4.1: Manuscript Preparation
- [ ] **PASS/FAIL**: Two manuscript drafts completed ("Generative Atlas", "Temporal Evolution")
- [ ] **PASS/FAIL**: All publication-quality figures exported to `figures/` directory
- [ ] **PASS/FAIL**: Manuscripts in shared document (Overleaf) with proper formatting
- [ ] **PASS/FAIL**: Key results tables generated from final model outputs

#### ✅ S4.4.2: Interoperability Documentation
- [ ] **PASS/FAIL**: README.md updated with "Using Results Outside Python" section
- [ ] **PASS/FAIL**: Data schemas documented comprehensively
- [ ] **PASS/FAIL**: Cross-language usage examples linked prominently
- [ ] **PASS/FAIL**: Documentation tested by non-Python user

#### ✅ S4.4.3: Artifact Publication
- [ ] **PASS/FAIL**: All artifacts uploaded to public repositories (GitHub/Zenodo/HuggingFace)
- [ ] **PASS/FAIL**: ONNX, CSV, Parquet files publicly accessible
- [ ] **PASS/FAIL**: Persistent DOIs obtained for data products
- [ ] **PASS/FAIL**: Clear organization and linking from main repository

#### ✅ S4.4.4: Final Dashboard Deployment
- [ ] **PASS/FAIL**: "Complete Discovery Platform" deployed to public URL
- [ ] **PASS/FAIL**: "Download Artifacts" section provides direct access to files
- [ ] **PASS/FAIL**: All demo components (Hierarchy Explorer, Zeitgeist Timeline) functional
- [ ] **PASS/FAIL**: Dashboard demonstrates cross-platform artifact portability

### SPRINT 4 FINAL VALIDATION
- [ ] **CRITICAL**: All scientific analyses completed and results logged
- [ ] **CRITICAL**: Two manuscripts drafted with complete figures/tables
- [ ] **CRITICAL**: All artifacts successfully exported to non-Python formats
- [ ] **CRITICAL**: Public dashboard deployed and fully functional
- [ ] **CRITICAL**: Cross-platform compatibility demonstrated

---

## DATA STRATEGY COMPLIANCE CRITERIA

### Universal Requirements (All Sprints)
These criteria apply to ALL sprints and must be validated in addition to sprint-specific criteria:

#### Development Phase Compliance
- [ ] **PASS/FAIL**: Development work uses appropriate data scale (`neurosynth_subset_1k` or specified subset)
- [ ] **PASS/FAIL**: CI/CD test cycles complete in <5 minutes using development data
- [ ] **PASS/FAIL**: All architectural features functional on development data before production transition
- [ ] **PASS/FAIL**: Development phase completed within specified timeframe (typically weeks 1-2)

#### Production Phase Compliance  
- [ ] **PASS/FAIL**: Production training uses complete `neurosynth_full_12k` dataset
- [ ] **PASS/FAIL**: Final demo validation performed with production-trained models
- [ ] **PASS/FAIL**: Pipeline compatibility verified during subset→full data transition
- [ ] **PASS/FAIL**: Production phase completed within specified timeframe (typically week 3)

#### Data Transition Validation
- [ ] **PASS/FAIL**: No loss of functionality during development→production data transition
- [ ] **PASS/FAIL**: Model architecture maintains compatibility across data scales
- [ ] **PASS/FAIL**: Performance metrics computed on production-scale data only
- [ ] **PASS/FAIL**: Production model checkpoints available for subsequent sprints

### Sprint-Specific Data Strategy Requirements

#### Sprint 1: Foundation & Baseline
- [ ] **PASS/FAIL**: Mock data → `neurosynth_subset_1k` transition completed by Week 1
- [ ] **PASS/FAIL**: First full-scale training run completed successfully on `neurosynth_full_12k`
- [ ] **PASS/FAIL**: "Latent Slider" demo validates with both development and production models

#### Sprint 2: Advanced Conditioning
- [ ] **PASS/FAIL**: All advanced features (FiLM, GRL, metadata imputation) validated on subset
- [ ] **PASS/FAIL**: Metadata distribution in subset adequate for conditioning validation
- [ ] **PASS/FAIL**: Final C-β-VAE training produces production-grade conditional model

#### Sprint 3: Precision & Uncertainty
- [ ] **PASS/FAIL**: Point-cloud and ensemble streams both use subset for development
- [ ] **PASS/FAIL**: Week 9 dual production training (Point-Cloud + Ensemble) completed
- [ ] **PASS/FAIL**: Uncertainty quantification valid only with full data distribution

#### Sprint 4: Synthesis & Release
- [ ] **PASS/FAIL**: Subset data officially retired - no development activities on subset
- [ ] **PASS/FAIL**: ALL activities use production models trained on `neurosynth_full_12k`
- [ ] **PASS/FAIL**: Released artifacts include data provenance documentation
- [ ] **PASS/FAIL**: Scientific claims supported exclusively by production-scale evidence

### Data Strategy Failure Criteria
- **IMMEDIATE SPRINT FAILURE**: Using wrong data scale for development vs production phases
- **IMMEDIATE SPRINT FAILURE**: Production metrics computed on development data
- **IMMEDIATE SPRINT FAILURE**: Public artifacts derived from development-scale models (Sprint 4)
- **IMMEDIATE SPRINT FAILURE**: Pipeline incompatibility during data scale transitions

---

## FAILURE RESPONSE PROTOCOL

If ANY sprint fails to meet ALL success criteria:

### 1. Immediate Assessment
- Document specific failed criteria with evidence
- Analyze root cause: technical, architectural, or resource constraints
- Estimate effort required to achieve success

### 2. Advanced Reasoning Engagement
- **Gemini Consultation**: Deep architectural analysis and alternative approaches
- **O3 Reasoning**: Systematic problem decomposition and solution validation
- **Cross-Model Validation**: Multiple AI perspectives on technical challenges

### 3. Success Path Planning
- Develop specific remediation plan with timeline
- Identify additional resources or expertise needed
- Set checkpoint reviews to prevent repeated failures

### 4. No Sprint Advancement
- **ABSOLUTE RULE**: Cannot proceed to next sprint until current sprint achieves 100% success
- Update progress_tracker.md with failure analysis and remediation plan
- Communicate status and revised timeline to stakeholders

This strict success criteria system ensures each sprint builds on solid foundations and that the final system meets all specified technical and scientific objectives.