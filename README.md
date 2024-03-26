# Comparative Analysis of Convolutional Neural Network and ScatNet for Brain Tumor Classification in Medical Images: A Study on Performance and Explainability

### Chiara Solito, Virginia Filippi, Alberto Righetti
The advent of deep learning has revolutionized many fields, including medical imaging, where deep techniques have shown remarkable performance in tasks like image classification, segmentation, and detection. However, despite their success, these models often act as black boxes, making their predictions difficult to interpret. This lack of transparency, known as the problem of explainability, is particularly concerning in the medical field, where understanding the reasoning behind a model’s prediction is crucial for trust and actionable insights.

We first provide a comparative study of Convolutional Neural Network (CNN) and ScatNet,  for image classification, in a medical imaging context: we will show you the results, comparing the performance of these two models to understand which is the best for the task under analysis. We also display and compare the filters extracted from the CNN (first convolutional layer) and the ScatNet.
Finally, we reduce the number of images until the ScatNet performs better than CNN.
For the XAI part, we introduce explainable algorithms, specifically Integrated Gradients and Local Interpretable Model-Agnostic Explanations (LIME), to shed light on the decision-making process of these models. We perform a statistical analysis on the final attributions, drawing conclusions on the methods implemented.\
By doing so, we hope to make these powerful tools more accessible and trustworthy for healthcare professionals, ultimately leading to better patient outcomes. 


## How to run
To analyze the performances of the trained models you can run the main, present in the folder.
Before that, be sure to have everything present in the requirements.txt installed. To create a conda environment:

```console
  $ conda create --name <env> --file requirements.txt
```

## Code organization

```console
  root
   |-- csv #contains training results, info about the splits train-validation
   |   |-- CNN 
   |   |-- norm
   |   |-- ScatNet
   |-- data -#image folder
   |   |-- test
   |   |   |-- meningioma
   |   |   |-- notumor
   |   |-- train
   |   |   |-- meningioma
   |   |   |-- notumor
   |-- models_trained
   |   |-- CNN 
   |   |-- images #training results plot, filters images, etc. 
   |   |-- ScatNet
   |-- report #contains report tex file and pdf
   |-- src #contains python scripts for training and utils
   | main.ipynb
   | README.md
   | requirements.txt
```


## Authors

- [@ChiaraSolito](https://github.com/ChiaraSolito)
- [@VirginiaFilippi](https://github.com/VirginiaFilippi)
- Alberto Righetti
