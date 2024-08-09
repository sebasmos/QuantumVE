## Q-Net

Quantum Classifier Vector Embedding framework using pretrained Neural network 


This work has received funding from the Google Cloud Research.



### Datasets used 

- [ABGQI-CNN dataset](https://zenodo.org/records/6027024): 8914 labeled sounds samples for 5 classes -  (7814) training, (850) validation, (850) testing, and 8914 total number of files.
- [Urban8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html): 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes

### Code structure

* [Vector embeddings extraction](part1_VE_extraction)
* [Models for Classic classification](part2_Classic_classification)
* [Models for quantum classification](part3_qubit_classification)


#### unsorted todo's

1. **Feature Extraction Diversification**: Utilize existing PyTorch models to extract feature embeddings at various sizes to enrich feature representation diversity.

2. **Quality Assessment of Vector Embeddings**: Implement rigorous quality checks for vector embeddings through methods such as linear probing or classifier evaluation to ensure optimal performance.

3. **Hilbert Space Adaptation via Quantum Circuit**: Integrate quantum circuit techniques to seamlessly adapt feature embeddings to Hilbert space, enhancing their representational capacity and facilitating advanced data processing capabilities.


Others: 


1. **Hyperparameter Tuning for Support Vector Classifier (SVC)**: Conduct comprehensive hyperparameter tuning for SVC across various kernels and parameter settings to optimize model performance.

2. **Feature Extraction from Pretrained Models**: Extract vector embeddings (VE) from pretrained models trained on spectrogram datasets, specifically the MAE Audio dataset, to leverage rich feature representations.

3. **Evaluation of Image Processing Transformations**: Assess the impact of additional image processing transformations on pretrained models sourced from ImageNet, especially those not readily available via standard TorchVision models repository, to address suboptimal results and determine if these extra processing steps are necessary for improved performance.

4. **Exploration of Alternative Quantum Neural Networks (QNN) and Torch Connectors**: Investigate alternative QNN architectures and Torch connectors to enhance model versatility and compatibility, ensuring robustness and adaptability across various tasks and datasets.

5. **define method for correct feature shape before converting to hilbert Space**


### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
