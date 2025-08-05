[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/QuantumVE/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/QuantumVE/) 
[![arXiv](https://img.shields.io/badge/arXiv-2508.00024-b31b1b.svg)](https://arxiv.org/abs/2508.00024)

# QuantumVE: Quantum-Transformer Advantage Boost Over Classical ML

**Breaking Discovery:** Vision Transformer embeddings unlock quantum machine learning advantage! First systematic proof that embedding choice determines quantum kernel success, revealing fundamental synergy between transformer attention and quantum feature spaces.

- 📂 **GitHub Repository**: [QuantumVE](https://github.com/sebasmos/QuantumVE)
- 📄 **Research Paper**: [Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning](https://arxiv.org/abs/2508.00024)
- 💻 **Dataset on HuggingFace**: [QuantumEmbeddings](https://huggingface.co/datasets/sebasmos/QuantumEmbeddings)

## 🎯 Breakthrough Results

- **8.02%** accuracy improvement on Fashion-MNIST vs classical SVMs
- **4.42%** boost on MNIST dataset  
- **First evidence** that ViT embeddings enable quantum advantage while CNN features show degradation
- **16-qubit** tensor network simulation via cuTensorNet proving scalability
- **Class-balanced k-means distillation** for efficient quantum processing

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

QuantumVE is **free** and **open source**, released under the [MIT License](https://github.com/sebasmos/QuantumVE/blob/main/LICENSE).

## 📚 Citation

### Paper
```bibtex
@article{Cajas2024_QuantumVE,
  title={Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning},
  author={Cajas Ordóñez, Sebastián Andrés and Torres Torres, Luis and Bifulco, Mario and Duran, Carlos and Bosch, Cristian and Simón Carbajo, Ricardo},
  journal={arXiv preprint arXiv:2508.00024},
  year={2024},
  url={https://arxiv.org/abs/2508.00024}
}
```

### Software
```bibtex
@software{Cajas2025_QSVM,
  author = {Cajas Ordóñez, Sebastián Andrés and Torres Torres, Luis and Bifulco, Mario and Duran, Carlos and Bosch, Cristian and Simón Carbajo, Ricardo},
  license = {MIT},
  month = {apr},
  title = {{Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning}},
  url = {https://github.com/sebasmos/QuantumVE},
  year = {2025}
}
```

---

<div align="center">

**🌟 Star us on GitHub if this helps your research! 🌟**

</div>
