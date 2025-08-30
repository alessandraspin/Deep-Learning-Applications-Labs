# Deep Learning Applications Laboratories
This repository contains the solutions of the semester exercises in which students put into practice the concepts and tools learned about Deep Learning.

## LAB1
**Topic**: Residual Networks, MLPs and CNNs for Image Classification.

**Exercise**:
* Implement simple and residual MLPs to classify MNIST digits and verify how residual connections improve training stability and accuracy on deeper networks.
* Experiment with CNNs on CIFAR-10, including deeper architectures and pre-trained ResNet models.
* Explore transfer learning and fine-tuning on CIFAR-100 using pre-trained models as feature extractors and applying different training strategies.

[Full details here](LAB1/README.md)

## LAB3
**Topic**: Transformer Models for Text and Image Classification.

**Exercise**:
* Use DistilBERT for Sentiment Analysis: establish a baseline with feature extraction and Machine Learning classifiers, then fine-tune for improved performance.
* Work with CLIP for image classification: evaluate zero-shot performance and fine-tune using parameter-efficient methods (LoRA) on Tiny ImageNet.
* Understand token preprocessing, model adaptation, and HuggingFace Trainer workflow.

[Full details here](LAB3/README.md)

## LAB4
**Topic**: Out-of-Distribution Detection and Adversarial Training.

**Exercise**:
* Build pipelines for detecting OOD samples using classifier confidence scores and autoencoder reconstruction error.
* Implement FGSM to generate adversarial examples and analyze their impact on model predictions.
* Study adversarial training and targeted attacks to explore the trade-off between adversarial robustness and OOD detection.

[Full details here](LAB4/README.md)
