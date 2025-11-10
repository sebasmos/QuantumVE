[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/QuantumVE/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.00024-b31b1b.svg)](https://arxiv.org/abs/2508.00024)

**Accepted for Poster, Presentation & Proceedings at:**
**3rd International Workshop on AI for Quantum and Quantum for AI (AIQxQIA 2025)**, ECAI 2025, Bologna, Italy, 25–30 October 2025

# Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning

**🚀 Key Discovery:**
Vision Transformer (ViT) embeddings unlock quantum machine learning advantage. This is the **first systematic evidence** that the choice of embeddings determines quantum kernel success, showing a fundamental synergy between transformer attention and quantum feature spaces.

### 🔗 Project Resources

* **Project Page:** [Embedding-Aware Quantum](https://sebasmos.github.io/embedding-aware-quantum/)
* **GitHub Repository:** [QuantumVE](https://github.com/sebasmos/QuantumVE)
* **Research Paper:** [Embedding-Aware Quantum-Classical SVMs](https://arxiv.org/abs/2508.00024)
* **Dataset on HuggingFace:** [QuantumEmbeddings](https://huggingface.co/datasets/sebasmos/QuantumEmbeddings)
* **Interactive Demo:** [Colab Notebook](https://github.com/sebasmos/QuantumVE/blob/main/Demo.ipynb)

### 🎯 Breakthrough Results

* **Fashion-MNIST:** +8.02% accuracy vs classical SVM
* **MNIST:** +4.42% accuracy boost
* **Embedding Insights:** ViT embeddings enable quantum advantage; CNN features degrade performance
* **Scalability:** 16-qubit tensor network simulation via cuTensorNet
* **Efficiency:** Class-balanced k-means distillation for quantum data preprocessing


## Project Architecture

```
QuantumVE/
├── data_processing/     # Class-balanced k-means distillation procedures
├── embeddings/          # Vision Transformer & CNN embedding extraction
├── qve/                 # Core quantum-classical modules and utilities
└── scripts/             # Experimental pipelines with cross-validation
    ├── classical_baseline.py           # Traditional SVM benchmarks
    ├── cross_validation_baseline.py    # Cross-validation framework
    └── qsvm_cuda_embeddings.py         # Our embedding-aware quantum method
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n QuantumVE python=3.11 -y
conda activate QuantumVE

# Clone and install
git clone https://github.com/sebasmos/QuantumVE.git
cd QuantumVE
pip install -e .

# For Ryzen devices - Install MPI
conda install -c conda-forge mpi4py openmpi
```

### 2. Download Pre-computed Embeddings

**MNIST Embeddings:**
```bash
mkdir -p data && \
wget https://huggingface.co/datasets/sebasmos/QuantumEmbeddings/resolve/main/mnist_embeddings.zip && \
unzip mnist_embeddings.zip -d data && \
rm mnist_embeddings.zip
```

**Fashion-MNIST Embeddings:**
```bash
mkdir -p data && \
wget https://huggingface.co/datasets/sebasmos/QuantumEmbeddings/resolve/main/fashionmnist_embeddings.zip && \
unzip fashionmnist_embeddings.zip -d data && \
rm fashionmnist_embeddings.zip
```

### 3. Run Experiments

**Single Node:**
```bash
# Classical baseline with cross-validation
python scripts/classical_baseline.py

# Cross-validation framework  
python scripts/cross_validation_baseline.py

# Our embedding-aware quantum method
python scripts/qsvm_cuda_embeddings.py
```

**Multi-Node with MPI:**
```bash
# Run with 2 processes
mpirun -np 2 python scripts/qsvm_cuda_embeddings.py
mpirun -np 2 python scripts/cross_validation_baseline.py
```

## 🔬 What Makes This Work?

Our key insight: **embedding choice is critical for quantum advantage**. While CNN features degrade in quantum systems, Vision Transformer embeddings create a unique synergy with quantum feature spaces, enabling measurable performance gains through:

1. **Class-balanced distillation** reduces quantum overhead while preserving critical patterns
2. **ViT attention mechanisms** align naturally with quantum superposition states
3. **Tensor network simulation** scales to practical problem sizes (16+ qubits)

## 🤝 Contributing

We welcome contributions! Help us advance quantum machine learning:

1. Fork the [QuantumVE repository](https://github.com/sebasmos/QuantumVE)
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Submit a pull request with detailed description

**Areas for contribution:**
- New embedding architectures (BERT, CLIP, etc.)
- Additional quantum backends
- Performance optimizations
- Documentation improvements

## 🙏 Acknowledgements

This work was supported by the **Google Cloud Research Credits program** under award number **GCP19980904**.

## 📄 License

[CC BY-NC-SA 4.0](https://github.com/sebasmos/QuantumVE/blob/main/LICENSE)

## 📚 Citation

### Paper
```bibtex
@inproceedings{ordonez2025embedding,
  title={Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning},
  author={Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s Cajas and Torres, Luis Fernando Torres and Bifulco, Mario and Duran, Carlos Andres and Bosch, Cristian and Carbajo, Ricardo Simon},
  booktitle={3rd International Workshop on AI for Quantum and Quantum for AI (AIQxQIA 2025), ECAI 2025},
  year={2025},
  month={October},
  address={Bologna, Italy},
  note={Accepted for Poster, Presentation \& Proceedings},
  url={https://arxiv.org/abs/2508.00024}
}
```

---

<div align="center">

**🌟 Star us on GitHub if this helps your research! 🌟**

</div>
