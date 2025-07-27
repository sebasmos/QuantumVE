[[LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/QuantumVE/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/QuantumVE/) 

# Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning

> 📩 **The full code will be made publicly available upon paper acceptance.**
> 🔒 The code is currently archived under embargo at Zenodo: [https://zenodo.org/records/15264568](https://zenodo.org/records/15264568). Access is available upon request for review purposes.

- 📂 **GitHub Repository**: [QuantumVE](https://github.com/sebasmos/QuantumVE)
- 💻 **Dataset on HuggingFace**: [*online*](https://huggingface.co/datasets/sebasmos/QuantumEmbeddings)

## Project Structure

- 📁 `Data Processing/`
- 📁 `Embeddings/`
- 📁 `Scripts/`
  - `cross_validation_baseline.py`
  - `qsvm_baseline_cuda_opt.py`
  - `qsvm_vector_embeddings.py`
  - `qsvm_heldout_test.py`
- 📁 `Figures/`

## Download MNIST or FashionMNIST Embeddings

```bash
mkdir -p data && \
wget https://huggingface.co/datasets/sebasmos/QuantumEmbeddings/resolve/main/mnist_embeddings.zip && \
unzip mnist_embeddings.zip -d data && \
rm mnist_embeddings.zip
```

> 🔁 FashionMNIST embeddings can be downloaded similarly from the same HuggingFace repository.

```bash
mkdir -p data && \
wget https://huggingface.co/datasets/sebasmos/QuantumEmbeddings/resolve/main/fashionmnist_embeddings.zip && \
unzip fashionmnist_embeddings.zip -d data && \
rm fashionmnist_embeddings.zip
```

## Setting Up Your Environment

1. **Create a Conda Environment:**
   ```bash
   conda create -n QuantumVE python=3.11.11 -y
   conda activate QuantumVE
   ```

2. **Install Dependencies**
   ```bash
   git clone https://github.com/sebasmos/QuantumVE.git
   cd QuantumVE
   pip install -e .
   ```

3. **[Ryzen Devices]** Install MPI via conda:
   ```bash
   conda install -c conda-forge mpi4py openmpi
   ```
   Ensure `mpi4py` is version `4.0.3`.

## Run on MPI - Multiple Nodes

```bash
mpirun -np 2 python test.py
mpirun -np 2 python cross_validation_baseline.py
```

## Contributing to QuantumVE

We welcome contributions! Start by forking the [QuantumVE repository](https://github.com/sebasmos/QuantumVE), and submit your enhancements via a pull request. All contributors will be credited in release notes.

Bug reports, new features, project ideas, or any improvements are highly encouraged. Help us grow the QuantumVE community!

## Acknowledgements

This work was supported by the Google Cloud Research Credits program under the award number GCP19980904.

## License

QuantumVE is **free** and **open source**, released under the [MIT License](https://github.com/sebasmos/QuantumVE/blob/main/LICENSE).

## Please Cite as

```
@software{Cajas2025_QSVM,
  author = {Cajas Ordóñez, Sebastián Andrés and Torres Torres, Luis and Bifulco, Mario and Duran, Carlos and Bosch, Cristian and Simón Carbajo, Ricardo},
  license = {MIT},
  month = apr,
  title = {{Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning}},
  url = {https://github.com/sebasmos/QuantumVE},
  year = {2025}
}
```
