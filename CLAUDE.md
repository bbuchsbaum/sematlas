# Generative, Hierarchical Atlas of Human Brain Function

## Project Overview

This project implements a state-of-the-art deep generative modeling framework to learn latent representations of human brain function from coordinate-based meta-analytic data. The system creates a dynamic, hierarchical, and rigorously validated generative atlas that transcends traditional linear meta-analysis limitations.

## Project Documentation Ecosystem

This CLAUDE.md serves as the central implementation guide, but the complete project knowledge is distributed across several critical documents:

### Core Documentation Files

#### **proposal.md** - Scientific Vision & Technical Specification
- **Purpose**: High-level scientific objectives and complete technical specification (Version 4.0)
- **Content**: Executive summary, model architectures, evaluation metrics, reproducibility requirements
- **When to consult**: Understanding project goals, technical approach, and scientific rationale
- **Key sections**: Core objectives, data pipeline design, model specifications, evaluation protocols

#### **SUCCESS_MARKERS.md** - Sprint Completion Criteria
- **Purpose**: Strict, binary success criteria for each sprint with failure protocols
- **Content**: PASS/FAIL criteria for every ticket, sprint validation requirements, remediation protocols
- **When to consult**: Before starting any ticket, validating completion, sprint transitions
- **Critical rule**: 100% success required before sprint advancement - no exceptions

#### **progress_tracker.md** - Live Project Status
- **Purpose**: Authoritative record of current status, completed work, and next actions
- **Content**: Current sprint/phase, ticket status, artifacts, decisions, blockers
- **When to consult**: Resuming work, understanding current state, tracking progress
- **Update frequency**: After every ticket completion and daily during active development

#### **Appendix1.md** - External Tools Technical Guide
- **Purpose**: Detailed technical overview of Neurosynth, NiMARE, and NeuroVault integration
- **Content**: Tool specifications, usage patterns, technical considerations, implementation details
- **When to consult**: Implementing data pipeline, debugging tool integration, understanding constraints
- **Key sections**: A1.2 (Neurosynth), A1.3 (NiMARE), A1.4 (NeuroVault), "Voxel-to-Insight Pipeline"

#### **sprint[1-4].md** - Detailed Sprint Implementation
- **Purpose**: Granular ticket specifications and sprint-specific requirements
- **Content**: Epic breakdowns, ticket descriptions, acceptance criteria, dependencies
- **When to consult**: Daily implementation work, understanding ticket requirements

### External Tools Overview (Detailed in Appendix1.md)

#### **Neurosynth** - Foundational Data Source
- ~12,000 fMRI studies with coordinates and metadata
- Coordinates already preprocessed to MNI152 space by Neurosynth
- Sparse metadata requiring robust imputation strategies
- **Implementation tickets**: S1.1.1 (download), S1.1.3 (coordinate validation)

#### **NiMARE** - Preprocessing Engine  
- Comprehensive neuroimaging meta-analysis toolkit
- Handles coordinate transformations, kernel convolutions, dataset management
- Critical for data pipeline robustness and reproducibility
- **Implementation tickets**: S1.1.1-S1.1.4 (entire data pipeline)

#### **NeuroVault** - Evaluation Ground Truth
- Repository of full 3D statistical brain maps
- Source of external validation data for generative quality assessment
- Enables evaluation beyond sparse coordinate patterns
- **Implementation tickets**: S2.2.4 (evaluation metrics), S4+ (final validation)

### Navigation Guidelines

1. **Starting work**: Read progress_tracker.md → current sprint docs → relevant Appendix1.md sections
2. **Implementing tickets**: Sprint docs → SUCCESS_MARKERS.md criteria → Appendix1.md technical details
3. **Completing tickets**: Update progress_tracker.md → validate against SUCCESS_MARKERS.md
4. **Sprint transitions**: Check SUCCESS_MARKERS.md completion → update progress_tracker.md
5. **Understanding failures**: SUCCESS_MARKERS.md protocols → advanced AI consultation → progress_tracker.md logging

### Core Scientific Objectives

1. **Learn a Disentangled & Hierarchical Latent Space**: Compress ~12,000 fMRI studies into a low-dimensional space where axes correspond to coherent neurocognitive processes
2. **Achieve High-Fidelity & Calibrated Generation**: Synthesize realistic, anatomically precise brain activation maps with well-calibrated uncertainty estimates
3. **Model & Mitigate Systematic Bias**: Actively model and de-bias for publication trends, scanner technology, and methodological choices
4. **Map Temporal Dynamics of Neuroscience**: Create a predictive model of scientific evolution through the learned latent space over time

## System Architecture

### Core Models

#### 1. Conditional β-VAE (3D CNN) - Primary Model
- **Backbone**: 3D ResNet-10 with dilated convolutions for >150mm receptive field
- **Normalization**: Group Normalization (groups=8) for stable training
- **Conditioning**: FiLM layers in both encoder and decoder
- **De-biasing**: Gradient Reversal Layer with adversarial year predictor
- **Purpose**: Main volumetric brain map generation with metadata conditioning

#### 2. Point-Cloud C-VAE - Precision Model
- **Architecture**: PointNet++ backbone with MLP decoder
- **Input**: Normalized to 30 points via probabilistic duplication/dropout
- **Loss**: Weighted combination of Chamfer Distance and Sinkhorn EMD
- **Features**: Gaussian Random Fourier Features for positional encoding
- **Purpose**: High-precision coordinate-level generation preserving millimeter-scale geometry

#### 3. Hierarchical VAE - Hierarchy Model
- **Structure**: Two-level hierarchy with protected skip paths
- **Constraints**: Gradient-stopped skip connections to prevent trivial identity mapping
- **Purpose**: Learn natural cognitive hierarchy and provide interpretable latent organization

### Data Pipeline Components

#### Input Data Sources
- **Neurosynth**: ~12,000 fMRI studies with coordinates and metadata
- **NiMARE**: Preprocessing and coordinate transformation toolkit
- **NeuroVault**: Ground-truth validation maps for evaluation

#### Multi-Scale Data Strategy
- **Development Data**: `neurosynth_subset_1k` - 1,000 study subset for rapid development and debugging
- **Production Data**: `neurosynth_full_12k` - Complete dataset for final model training and validation
- **Mock Data**: 100 synthetic studies for initial pipeline testing (development only)
- **Transition Protocol**: Explicit development→production phases in each sprint

#### Preprocessing Pipeline
- **Coordinate Standardization**: tal2icbm transformation with RMSD logging
- **Directional Deduplication**: Hash-based deduplication preserving statistical contrast direction
- **Volumetric Generation**: Dual-kernel (6mm/12mm) Gaussian convolution with orientation augmentation
- **Metadata Imputation**: Uncertainty-aware imputation with amortization heads

## Technical Stack

### Core Dependencies
```yaml
# Deep Learning
torch>=1.12.0
pytorch-lightning>=1.6.0
torchvision>=0.13.0

# Neuroimaging
nimare>=0.0.14
nibabel>=4.0.0
nilearn>=0.9.0

# Data Science
numpy>=1.21.0
pandas>=1.4.0
scipy>=1.8.0
scikit-learn>=1.1.0

# Visualization
matplotlib>=3.5.0
plotly>=5.8.0
streamlit>=1.10.0

# Storage & Performance
lmdb>=1.3.0
h5py>=3.7.0
dvc>=2.10.0

# Monitoring
wandb>=0.12.0
```

### Hardware Requirements
- **Training**: NVIDIA GPU with ≥16GB VRAM (RTX 4090 or A6000 recommended)
- **Memory**: ≥32GB RAM for data loading
- **Storage**: ≥100GB for cached data and model checkpoints

## Development Plan: 4 Sprint Structure

### Sprint 1: Data Foundation & Baseline Model (3 weeks)
**Goal**: Establish versioned data pipeline and baseline 3D VAE
**Key Demo**: "Latent Slider" - Interactive latent space traversal

#### Epic 1: Data Curation Pipeline
- `S1.1.1`: Setup Neurosynth download with NiMARE
- `S1.1.2`: Implement directional deduplication logic
- `S1.1.3`: Validate coordinate space (Neurosynth preprocessing verification)
- `S1.1.4`: Create volumetric cache with dual-kernel augmentation
- `S1.1.5`: Finalize splits and DVC versioning

#### Epic 2: Baseline 3D VAE
- `S1.2.1`: Define ResNet VAE architecture with Group Normalization
- `S1.2.2`: Create PyTorch Lightning DataModule with LMDB loading
- `S1.2.3`: Create Lightning Module with VAE loss and reparameterization
- `S1.2.4`: Setup Weights & Biases training script

#### Epic 3: Interactive Demo
- `S1.3.1`: Create model inference wrapper with checkpoint loading
- `S1.3.2`: Build Jupyter notebook with ipywidgets and nilearn viewer

### Sprint 2: Advanced Conditioning & Architecture (3 weeks)
**Goal**: Upgrade to Conditional β-VAE with FiLM and adversarial de-biasing
**Key Demo**: "Counterfactual Machine" - Conditional generation dashboard
**Infrastructure**: RunPod GPU platform for cloud training (migrated from Paperspace)

#### Epic 0: RunPod GPU Setup
- `S2.0.1`: Setup RunPod SDK and authentication
- `S2.0.2`: Create console-based training orchestration scripts
- `S2.0.3`: Configure code/environment synchronization
- `S2.0.4`: Integrate W&B logging with cloud training

#### Epic 1: Advanced Architecture
- `S2.1.1`: Upgrade backbone to DenseNet with dilated convolutions
- `S2.1.2`: Implement metadata imputation with amortization head
- `S2.1.3`: Implement FiLM conditioning layers
- `S2.1.4`: Implement GRL adversarial de-biasing

#### Epic 2: Training Hardening
- `S2.2.1`: Update DataModule with metadata conditioning
- `S2.2.2`: Implement KL divergence controller callback
- `S2.2.3`: Refine optimizer to AdamW with β₂=0.995
- `S2.2.4`: Implement formal evaluation metrics

#### Epic 3: Conditional Demo
- `S2.3.1`: Upgrade inference wrapper for conditional generation
- `S2.3.2`: Build Streamlit/Dash conditional dashboard

### Sprint 3: Precision & Uncertainty (3 weeks)
**Goal**: Parallel development of Point-Cloud VAE and Deep Ensemble uncertainty
**Key Demo**: "Dual-View Confidence Explorer" - Geometry + uncertainty visualization

#### Epic 1: Point-Cloud VAE (Stream A)
- `S3.1.1`: Create HDF5 point-cloud cache
- `S3.1.2`: Implement PointNet++ VAE architecture
- `S3.1.3`: Integrate metadata conditioning for point clouds
- `S3.1.4`: Develop point-cloud trainer with Chamfer+EMD loss
- `S3.1.5`: Build 3D scatter plot viewer for reconstructions

#### Epic 2: Deep Ensemble Uncertainty (Stream B)
- `S3.2.1`: Modify decoder for aleatoric uncertainty (μ, log σ²)
- `S3.2.2`: Implement snapshot ensembling with cyclical LR
- `S3.2.3`: Implement Expected Calibration Error (ECE) metric
- `S3.2.4`: Create ensemble inference wrapper
- `S3.2.5`: Build confidence explorer with uncertainty layers

### Sprint 4: Synthesis & Public Release (3 weeks)
**Goal**: Scientific analysis, manuscript preparation, and interoperable release
**Key Demo**: "Complete Discovery Platform" - Full integrated system

#### Epic 1: Cross-Model Analysis
- `S4.1.1`: Run Canonical Correlation Analysis between models
- `S4.1.2`: Train H-VAE prototype with stage-gate validation
- `S4.1.3`: Run HDBSCAN clustering on latent space

#### Epic 2: Temporal Analysis
- `S4.2.1`: Create chronological latent time series
- `S4.2.2`: Train trajectory GRU for forecasting
- `S4.2.3`: Build "Neuroscience Zeitgeist" timeline

#### Epic 3: Interoperable Release
- `S4.3.1`: Export tabular data to CSV and Parquet
- `S4.3.2`: Export models to ONNX format
- `S4.3.3`: Create non-Python usage examples
- `S4.3.4`: Standardize outputs to NIfTI format

#### Epic 4: Documentation & Deployment
- `S4.4.1`: Write manuscripts and generate figures
- `S4.4.2`: Update documentation for interoperability
- `S4.4.3`: Publish all artifacts to public repositories
- `S4.4.4`: Deploy final dashboard with download links

## Data Strategy Addendum

### Overview
The project implements a **dual-scale data strategy** that balances development velocity with production rigor. Each sprint is divided into explicit **Development** and **Production** phases, using appropriately sized datasets for each purpose.

### Data Scale Definitions
1. **`neurosynth_subset_1k`**: 1,000 study subset for development, debugging, and rapid iteration
2. **`neurosynth_full_12k`**: Complete ~12,000 study dataset for production validation and final models
3. **Mock data**: 100 synthetic studies for initial pipeline development (Sprint 1 only)

### Phase Structure
Each sprint follows a consistent **Development → Production** progression:

#### Development Phase (Weeks 1-2 of each sprint)
- **Purpose**: Rapid iteration, debugging, architecture development
- **Data**: `neurosynth_subset_1k` exclusively
- **Benefits**: Fast CI/CD cycles (<5 minutes), low compute costs, rapid failure detection
- **Activities**: New feature implementation, unit testing, integration debugging

#### Production Phase (Final week of each sprint)
- **Purpose**: Final validation, production-grade model training, demo creation
- **Data**: `neurosynth_full_12k` exclusively
- **Benefits**: Scale validation, realistic performance metrics, production artifacts
- **Activities**: Full-scale training runs, final model checkpoints, demo validation

### Sprint-Specific Data Strategy

#### Sprint 1: Foundation & Baseline
- **Development (Weeks 1-2)**: Mock data → `neurosynth_subset_1k` transition for pipeline development
- **Production (Week 3)**: First full-scale training run on `neurosynth_full_12k` for baseline model

#### Sprint 2: Advanced Conditioning
- **Development (Weeks 4-5)**: All architectural features developed on `neurosynth_subset_1k`
- **Production (Week 6)**: Definitive C-β-VAE training on `neurosynth_full_12k`

#### Sprint 3: Precision & Uncertainty
- **Development (Weeks 7-8)**: Parallel streams on subset data (point-cloud and ensemble prototypes)
- **Production (Week 9)**: Two major production runs (Point-Cloud C-VAE and Deep Ensemble)

#### Sprint 4: Synthesis & Release
- **Production Exclusivity (Weeks 10-12)**: Subset data officially retired, all activities use full-scale models

### Implementation Requirements
1. **CI/CD Configuration**: Test pipelines configured for subset data (<5 min runtime)
2. **Data Pipeline Validation**: Stress-testing at full scale before production use
3. **Checkpoint Management**: Clear versioning for development vs production models
4. **Resource Planning**: Cloud GPU allocation aligned with production phase timing

### Quality Assurance
- **Development Validation**: Functional correctness on subset data
- **Production Validation**: Scale performance and final quality metrics
- **Transition Verification**: Pipeline compatibility across data scales
- **Cost Management**: Subset development minimizes compute costs during iteration

This strategy ensures rapid development cycles while maintaining production quality and provides clear checkpoints for scale validation throughout the project.

## Progress Tracking & Continuity

### Critical Requirement: Maintain progress_tracker.md
**MANDATORY**: The `progress_tracker.md` file MUST be kept up-to-date throughout the entire project. This file serves as the authoritative record of:
- Current sprint and phase
- Completed tickets with timestamps and validation status
- Active tickets with current status and blockers
- Next immediate tasks and dependencies
- Key artifacts and their locations
- Critical decisions and their rationale

### Progress Tracking Protocol
1. **After Each Ticket Completion**: Update progress_tracker.md with completion status, artifacts produced, and validation results
2. **Daily Status Updates**: Log current work status, blockers encountered, and next steps
3. **Sprint Transitions**: Document sprint completion status against SUCCESS_MARKERS.md criteria
4. **Context Preservation**: Ensure sufficient detail that any team member (or Claude instance) can resume work immediately

### Resume Protocol
When resuming work after interruption:
1. Read progress_tracker.md to understand current state
2. Review SUCCESS_MARKERS.md for current sprint criteria (strict PASS/FAIL validation)
3. Validate last completed ticket status against success markers
4. Identify next priority task from current sprint
5. Update progress_tracker.md with resume timestamp

### Success Validation Protocol
All work must be validated against SUCCESS_MARKERS.md:
- **Binary validation**: Every criterion must be PASS - no partial credit
- **Sprint gates**: 100% success required before advancement
- **Failure response**: Engage advanced AI (Gemini, O3) for problem resolution
- **Documentation**: All decisions and validations logged in progress_tracker.md

This progress tracking system ensures project continuity across context switches, team changes, and extended development periods while maintaining strict quality standards.

## Implementation Guidelines

### Training Procedures
- **Optimizer**: AdamW with β₂=0.995 for KL stability
- **KL Annealing**: Dynamic controller with automatic β adjustment
- **Precision**: FP32 for point-cloud models, AMP for volumetric models
- **Ensemble**: 5-model snapshot ensemble with cyclical learning rates

### Data Integrity
- **Coordinate Validation**: RMSD ≤4mm post-transformation logging
- **Version Control**: DVC for full pipeline reproducibility
- **Quality Checks**: Automated validation of tensor shapes and value ranges
- **Scale Validation**: Pipeline compatibility testing across subset and full datasets
- **Data Strategy Compliance**: Strict adherence to development vs production data usage protocols

### Evaluation Metrics
- **Generative Quality**: Improved Precision & Recall for Distributions (PRD)
- **Classification**: Zero-shot linear probe on HCP task contrasts
- **Forecasting**: Fréchet Distance for temporal predictions
- **Calibration**: Expected Calibration Error (ECE) < 0.15

### Uncertainty Quantification
- **Epistemic**: Deep ensemble variance across K=5 models
- **Aleatoric**: Learned variance parameters in decoder
- **Calibration**: Temperature scaling if ECE > 0.1

## Key Deliverables

### Interactive Demonstrations
1. **Latent Slider**: Single-dimension latent space traversal
2. **Counterfactual Machine**: Conditional generation with metadata controls
3. **Dual-View Confidence Explorer**: Geometry + uncertainty visualization
4. **Complete Discovery Platform**: Integrated system with hierarchy and timeline

### Scientific Outputs
- Cross-model correlation analysis revealing shared latent dimensions
- Hierarchical cognitive organization from H-VAE
- Temporal evolution of neuroscience research patterns
- Uncertainty-calibrated brain map generation

### Public Release Artifacts
- **Models**: ONNX format for cross-platform compatibility
- **Data**: CSV/Parquet tabular exports
- **Images**: NIfTI standard format brain maps
- **Code**: Fully documented Python repositories
- **Documentation**: Non-Python usage examples

## Quality Assurance

### Testing Strategy
- Unit tests for all data processing functions
- Integration tests for model training pipelines
- Validation tests against known ground-truth data
- Performance benchmarks for inference speed

### Reproducibility Requirements
- `TORCH_DETERMINISTIC=1` for bit-wise reproducibility
- Pinned dependency versions in environment files
- DVC pipeline versioning for data processing
- Comprehensive logging of all hyperparameters

### Code Quality
- Type hints for all function signatures
- Docstrings following NumPy style
- Pre-commit hooks for code formatting
- Continuous integration with automated testing

## Getting Started

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd sematlas

# Setup environment
conda env create -f environment.yml
conda activate sematlas

# Initialize DVC and download data
dvc pull
dvc repro

# Start training baseline model
python train.py --config configs/baseline_vae.yaml
```

### Project Structure
```
sematlas/
├── data/                   # Raw and processed data
├── src/
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and callbacks
│   └── evaluation/        # Metrics and validation
├── configs/               # Training configurations
├── scripts/               # Utility scripts
├── notebooks/             # Interactive demos
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation and figures
```

This comprehensive specification provides the foundation for implementing a cutting-edge generative atlas of human brain function, with clear sprint-based development, rigorous evaluation, and broad accessibility through interoperable formats.