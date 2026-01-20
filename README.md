# Presidential AI Challenge: Diabetic Retinopathy Detection

This project presents a state-of-the-art deep learning solution for automated detection of diabetic retinopathy from retinal fundus images, developed for the Presidential AI Challenge. Our model achieves exceptional performance with 99.73% AUC and 98.28% accuracy using EfficientNet-B3 architecture optimized for Apple Silicon.

## 🏆 Key Achievements
- **99.73% AUC** - Near-perfect discrimination capability
- **98.28% Accuracy** - High diagnostic precision
- **Optimized for M4 Mac** - Efficient training on consumer hardware
- **Research-Grade Visualizations** - Comprehensive evaluation metrics
- **Medical AI Best Practices** - Robust validation and explainability

## 🚀 Model Architecture
- **Backbone**: EfficientNet-B3 (pre-trained on ImageNet)
- **Input**: 300x300 RGB fundus images
- **Output**: Binary classification (No DR vs DR)
- **Training Hardware**: M4 Mac Mini with Metal Performance Shaders

## 📊 Performance Metrics
- **AUC**: 0.9973
- **Accuracy**: 98.28%
- **Precision**: 98.28%
- **Recall**: 98.29%
- **F1-Score**: 98.28%
- **Test Dataset**: 3,662 fundus images

## 🔬 Advanced Training Techniques
- Focal Loss with label smoothing for class imbalance
- Mixed precision training (FP16) for memory efficiency
- Gradient accumulation for larger effective batch sizes
- Comprehensive data augmentation pipeline
- Early stopping with model checkpointing
- Cosine annealing learning rate schedule

## 🛠️ Quickstart
1. Clone this repo and install requirements:
   ```bash
   git clone <repository-url>
   cd fundus_disease_detection
   pip install -r requirements.txt
   ```
2. Download the APTOS dataset and organize as per `data/README.md`.
3. Run evaluation with pre-trained model:
   ```bash
   python evaluate.py --config configs/aptos_binary_m4.yaml --checkpoint "Binary Classification Model/best_model_auc_0.9971.pth copy"
   ```
4. Generate research visualizations:
   ```bash
   python generate_research_plots.py
   ```

## 📁 Project Structure
- `configs/` — YAML configuration files for reproducibility
- `utils/` — Dataset loaders, metrics, and helper functions
- `evaluate.py` — Model evaluation and confusion matrix generation
- `generate_research_plots.py` — Comprehensive visualization generator
- `train.py` — Training pipeline with advanced techniques
- `Binary Classification Model/` — Pre-trained model checkpoint
- `research_plots/` — Generated evaluation visualizations

## 📈 Evaluation Results
Run the evaluation script to generate:
- Confusion matrix with performance metrics
- ROC and Precision-Recall curves
- Prediction score distributions
- Sample predictions with confidence scores
- Performance metrics across thresholds

## ⚙️ Model Configuration
The model is optimized for Apple Silicon (M4 Mac) with:
- Metal Performance Shaders (MPS) backend
- Mixed precision training (FP16)
- Gradient accumulation for memory efficiency
- Custom data augmentation pipeline

## 🏥 Medical AI Considerations
- Rigorous validation on diverse dataset
- Explainability through Grad-CAM visualizations
- Ethical AI practices and bias mitigation
- Research-grade reproducibility

## 📄 License
For research and educational use only. See `LICENSE`.

## 🤝 Contributing
Open to contributions for improving diabetic retinopathy detection and medical AI applications.
