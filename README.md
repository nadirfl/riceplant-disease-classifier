# Rice Leaf Image Classifier

## Table of contents
1. Project goal and motivation
2. Data collection and description
3. Prerequisites
4. Project structure
5. Modeling and Training
6. Interpretation
7. Validation

TODO: add index for titles according to table of contents

## 1. Project goal and motivation
For this project, I wanted to build an application that could potentially solve a real-world problem. I was also keen on experimenting with Computer Vision and Image Classifiers more since these are fascinating concepts that I'd like to know more of.

After researching for existing datasets on Kaggle, I've stumpled upon the dataset for rice plant diseases. After reviewing the available images and documentation, I was certain to build a model with this dataset, since it's an applicable model for a real-world problem.

## 2. Data collection and description
Link to the dataset: https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset

*The Rice Life Disease Dataset is an extensive collection of data focused on three major diseases that affect rice plants: Bacterial Blight (BB), Brown Spot (BS), and Leaf Smut (LS).*

The dataset consists of 3 directories (as of 20th of May 2024):

    rice-plant-diseases-dataset
    ├── Bacterialblight: 1604 images
    ├── Brownspot:       1620 images
    └── Leafsmut:        1460 images

The images have a decent quality where each disease is visible and recognizable.

Definition of the rice plant diseases:
- Bacterial blight: deadly bacterial disease that is among the most destructive afflictions of cultivated rice. In severe epidemics, crop loss may be as high as 75 percent, and millions of hectares of rice are infected annually
- Brown spot: a fungal disease that infects the coleoptile, leaves, leaf sheath, panicle branches, glumes, and spikelets. Its most observable damage is the numerous big spots on the leaves which can kill the whole leaf
- Leaf smut: a widely distributed, but somewhat minor, disease of rice. The fungus produces slightly raised, angular, black spots (sori) on both sides of the leaves. Although rare, it also can produce spots on leaf sheaths

## 3. Prerequisites
There are no specific requirements to run this application. You need the usual installments for Java 21, etc. which we have already completed in the course "Model Deployment and Maintenance".

However, to train the model locally, you need to download the dataset which you can find [here](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset). I've downloaded the dataset on the 20th of May 2024.

After downloading and unzipping the dataset make sure to first rename the folder to "rice_leafs" and then place it under the root folder of the application (see chapter "Project structure" for the exact location).

## 4. Project structure
    RICEPLANTDISEASES
    |
    ├── models/                                     # Directory containing related files for the model
    |   ├── riceLeafClassifier-0002.params          # File containing the parameters for the image classifier
    |   └── synset.txt                              # File containing the labels for the image classifier
    |   |
    ├── rice_leafs/                                 # Directory containing the images for training and validation
    |   ├── Bacterialblight/                        # Directory containing images of rice plants with bacterial blight (BB)
    |   ├── Brownspot/                              # Directory containing images of rice plants with brown spot (BS)
    │   └── Leafsmut/                               # Directory containing images of rice plants with leaf smut (LS)
    │
    ├── src/main                                    # Directory containing the source code
    │   ├── java/ch/zhaw/deeplearningjava/ferlinad/riceplantdiseases
    │   │   ├── ClassifierController.java           # Class containing the defined endpoints for the application
    │   │   ├── Inference.java                      # Class containing the logic for the inference of the model
    │   │   ├── Models.java                         # Class containing the setup and configuration of the model
    |   |   ├── RiceplantdiseasesApplication.java   # Class containing the main method to start the application locally
    │   │   └── Training.java                       # Class containing the logic for the model training
    │   │
    |   ├── resources
    │   |   ├── static/
    │   |   │   ├── index.html                      # File containing the structure of the UI
    │   |   │   ├── script.js                       # File containing logic for backend calls
    │   |   └── application.properties              # File containing properties for the application
    |   
    ├── test_dataset                                # Directory consistinc of 15 images for model testing
    |
    ├── .gitignore                                  # File containing a list of excluded files/directories not to be pushed to GitHub
    ├── pom.xml                                     # File containing the dependencies
    └── README.md                                   # Project overview and documentation

## 5. Modeling and Training
For modelling I'm using the [DeepJavaLibrary](https://djl.ai/). This library allows me to quickly build and fine-tune a model and provides various neural networks for different problems. 

I'm using the Residual Network (ResNet) which is a generic implementation adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py (Original author Wei Wu) by Antti-Pekka Hynninen. This implements the original resnet ILSVRC 2015 winning network from Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun and is named as a "Deep Residual Learning for Image Recognition"

In the class ```Models.java```, I start off by building the first block of the network.
```java
Block resNet50 = ResNetV1.builder()
    .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
    .setNumLayers(NUM_OF_LAYERS)
    .setOutSize(NUM_OF_OUTPUT)
    .build();
```
The neural network consists of 50 layers and have 3 output layers (labels for the diseases)
The constants (IMAGE_HEIGHT, etc.) are defined in the class and have been altered for the different runs (see down below).

The training is done in the class ```Training.java``` where I use the dataset to train my model. To start the training you can start the ```main()``` method.


I've chosen the following (hyper-)parameters after experimenting (see test runs down below):
- Training split: 80/20
- Training configuration: 
  - Softmax Cross Entropy Loss function
    - MultiFactorTracker added to reduce learning rate after certain amount of epochs
- Image transformations
  - Resizing to 244x244
  - Random flip (left/right, top/bottom)
- Batch size before updating model: 32
- Epochs: 10

After training, the model (riceLeafClassifier-0002.params) and the labels (synset.txt) are saved in the directory /models.

### 5.1 First run
#### 5.1.1 Hyperparameters:
Model
- Layers: 50
- Image height and width: 100

Training
- Batch Size: 32
- Epochs: 2
- Split: 80/20
- Training configuration: loss
- evaluator: accuracy
- Transformations
  - ```.addTransform(new RandomFlipLeftRight())```
  - ```.addTransform(new RandomFlipTopBottom())```

#### 5.1.2 Results:
```
13:09:57.304 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
13:09:57.547 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
13:09:57.548 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.072 ms.
Training:    100% |========================================| Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.78
Validating:  100% |========================================|
13:16:04.351 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
13:16:04.353 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.78
13:16:04.354 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.77, SoftmaxCrossEntropyLoss: 1.11
Training:    100% |========================================| Accuracy: 0.85, SoftmaxCrossEntropyLoss: 0.53
Validating:  100% |========================================|
13:21:18.169 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
13:21:18.170 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.85, SoftmaxCrossEntropyLoss: 0.54
13:21:18.171 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.89, SoftmaxCrossEntropyLoss: 0.52
```
#### 5.1.3 Conclusion:
- Really good accuracy of 0.89 in the validation for the first run
- I'll try to improve this more

### 5.2 Second Run
#### 5.2.1 Changes
Training
- Transformations removed
  - ```.addTransform(new RandomFlipLeftRight())```
  - ```.addTransform(new RandomFlipTopBottom())```

#### 5.2.2 Results:
```
13:23:59.111 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
13:23:59.314 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
13:23:59.318 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.303 ms.
Training:    100% |========================================| Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.70
Validating:  100% |========================================|
13:29:38.131 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
13:29:38.132 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.79, SoftmaxCrossEntropyLoss: 0.69
13:29:38.132 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.91, SoftmaxCrossEntropyLoss: 0.25
Training:    100% |========================================| Accuracy: 0.89, SoftmaxCrossEntropyLoss: 0.39
Validating:  100% |========================================|
13:35:04.231 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
13:35:04.232 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.89, SoftmaxCrossEntropyLoss: 0.39
13:35:04.233 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.82, SoftmaxCrossEntropyLoss: 0.58
```

#### 5.2.3 Conclusion:
- Performance a bit worse (0.82 vs 0.89 accuracy). Keeping the transformations

### 5.3 Third Run:
#### 5.3.1 Changes
Model
- Image height and width: 224x224

Training
- Epochs: 10
- Transformations
  - added ```.addTransform(new RandomFlipLeftRight())```
  - added ```.addTransform(new RandomFlipTopBottom())```
- MultiFactorTracker added to reduce learning rate after certain amount of epochs

#### 5.3.2 Results
```
13:44:46.197 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
13:44:46.385 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
13:44:46.386 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.084 ms.
Training:    100% |========================================| Accuracy: 0.60, SoftmaxCrossEntropyLoss: 0.92
Validating:  100% |========================================|
14:06:06.442 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
14:06:06.452 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.60, SoftmaxCrossEntropyLoss: 0.92
14:06:06.452 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.40, SoftmaxCrossEntropyLoss: 1.20
Training:    100% |========================================| Accuracy: 0.66, SoftmaxCrossEntropyLoss: 0.82
Validating:  100% |========================================|
14:52:42.348 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
14:52:42.367 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.66, SoftmaxCrossEntropyLoss: 0.82
14:52:42.368 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.58, SoftmaxCrossEntropyLoss: 0.93
Training:    100% |========================================| Accuracy: 0.69, SoftmaxCrossEntropyLoss: 0.75
Validating:  100% |========================================|
15:11:44.218 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 3 finished.
15:11:44.234 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.69, SoftmaxCrossEntropyLoss: 0.75
15:11:44.235 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.69, SoftmaxCrossEntropyLoss: 0.76
Training:    100% |========================================| Accuracy: 0.72, SoftmaxCrossEntropyLoss: 0.68
Validating:  100% |========================================|
15:30:32.287 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 4 finished.
15:30:32.303 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.72, SoftmaxCrossEntropyLoss: 0.69
15:30:32.303 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.64, SoftmaxCrossEntropyLoss: 0.86
Training:    100% |========================================| Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.61
Validating:  100% |========================================|
15:50:51.980 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 5 finished.
15:50:51.996 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.62
15:50:51.997 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.66, SoftmaxCrossEntropyLoss: 0.89
Training:    100% |========================================| Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.58
Validating:  100% |========================================|
16:08:24.364 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 6 finished.
16:08:24.365 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.57
16:08:24.365 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.57
Training:    100% |========================================| Accuracy: 0.80, SoftmaxCrossEntropyLoss: 0.53
Validating:  100% |========================================|
16:26:25.269 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 7 finished.
16:26:25.286 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.80, SoftmaxCrossEntropyLoss: 0.53
16:26:25.287 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.72, SoftmaxCrossEntropyLoss: 0.75
Training:    100% |========================================| Accuracy: 0.82, SoftmaxCrossEntropyLoss: 0.49
Validating:  100% |========================================|
16:44:52.415 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 8 finished.
16:44:52.429 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.82, SoftmaxCrossEntropyLoss: 0.49
16:44:52.430 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.81, SoftmaxCrossEntropyLoss: 0.53
Training:    100% |========================================| Accuracy: 0.83, SoftmaxCrossEntropyLoss: 0.47
Validating:  100% |========================================|
17:02:56.259 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 9 finished.
17:02:56.274 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.83, SoftmaxCrossEntropyLoss: 0.47
17:02:56.274 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.59
Training:    100% |========================================| Accuracy: 0.83, SoftmaxCrossEntropyLoss: 0.46
Validating:  100% |========================================|
17:19:54.177 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 10 finished.
17:19:54.178 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.83, SoftmaxCrossEntropyLoss: 0.47
17:19:54.185 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.73, SoftmaxCrossEntropyLoss: 0.60
```

#### 5.3.3 Conclusion
- leaving notebook on standby was not beneficial for the training performance (computational)
- Training took a lot of time and did not yield better results
- Reducing the image size back to 100x100 and epoch to 2

### 5.4 Fourth Run
#### 5.4.1 Changes
- Epoch 2
- Image Size: 100 x 100
- Transformations
  - added ```.addTransform(new RandomResizedCrop(imageHeight, imageWidth))```
  - added ```.addTransform(new RandomBrightness(0.1f))```
- removed MultiFactorTracker

#### 5.4.2 Results
```
17:24:41.630 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
17:24:41.938 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
17:24:41.940 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.088 ms.
Training:    100% |========================================| Accuracy: 0.65, SoftmaxCrossEntropyLoss: 0.97
Validating:  100% |========================================|
17:30:02.229 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
17:30:02.231 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.65, SoftmaxCrossEntropyLoss: 0.96
17:30:02.232 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.73, SoftmaxCrossEntropyLoss: 0.67
Training:    100% |========================================| Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.59
Validating:  100% |========================================|
17:37:51.931 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
17:37:51.949 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.77, SoftmaxCrossEntropyLoss: 0.60
17:37:51.949 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.79, SoftmaxCrossEntropyLoss: 0.55
```
#### 5.4.3 Conclusion
- The performance is worse with the added transformations
- Restoring the parameters to the best model yet

### 5.5 Final Run
#### 5.5.1 Changes
- Transformations
  - removed ```.addTransform(new RandomResizedCrop(imageHeight, imageWidth))```
  - removed ```.addTransform(new RandomBrightness(0.1f))```
#### 5.5.2 Result
```
17:39:43.938 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
17:39:44.256 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
17:39:44.258 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.170 ms.
Training:    100% |========================================| Accuracy: 0.78, SoftmaxCrossEntropyLoss: 0.75
Validating:  100% |========================================|
17:44:51.478 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
17:44:51.480 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.79, SoftmaxCrossEntropyLoss: 0.75
17:44:51.481 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.93, SoftmaxCrossEntropyLoss: 0.27
Training:    100% |========================================| Accuracy: 0.91, SoftmaxCrossEntropyLoss: 0.28
Validating:  100% |========================================|
17:49:50.117 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
17:49:50.119 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.91, SoftmaxCrossEntropyLoss: 0.28
17:49:50.120 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.93, SoftmaxCrossEntropyLoss: 0.25
```
#### 5.5.3 Conclusion
With a validation accuracy of 0.93 this is the best result yet. I will keep the current model.

## 6. Interpretation
After trying out different parameters for training the model and evaluating the performances, I've decided to use the model from the first run (also last run in the training section above) since it yielded the best performance in the validation.

Here you can find an overview of the different test runs:

| Run nr. | Testing Accuracy | Testing Loss | Validation Accuracy | Validation Loss | Ellapsed training time |
| ------- | ---------------- | ------------ | ------------------- | --------------- | ---------------------- |
| 1       | 0.85             | 0.54         | 0.89                | 0.52            | 11min                  |
| 2       | 0.89             | 0.39         | 0.82                | 0.58            | 11min                  |
| 3       | 0.83             | 0.47         | 0.73                | 0.60            | **3h 30min**           |
| 4       | 0.77             | 0.60         | 0.79                | 0.55            | 13min                  |
| 5       | 0.79             | 0.75         | 0.93                | 0.27            | 10min                  |

## 7. Validation
For validating my model, I've created the directory /test_dataset consisting of 5 pictures per rice plant disease that I've manually downloaded from Google Images. Here are the results:

| Picture Name | Actual Disease  | Predicted Disease | Score (0-1) |
| ------------ | --------------  | ----------------- | ----------- |
| bb1.png      | Bacterialblight | Brownspot         | 0           |
| bb2.png      | Bacterialblight | Bacterialblight   | 1           |
| bb3.png      | Bacterialblight | Bacterialblight   | 1           |
| bb4.png      | Bacterialblight | Bacterialblight   | 1           |
| bb5.png      | Bacterialblight | Leafsmut          | 0           |
| bs1.png      | Brownspot       | Leafsmut          | 0           |
| bs2.png      | Brownspot       | Leafsmut          | 0           |
| bs3.png      | Brownspot       | Bacterialblight   | 0           |
| bs4.png      | Brownspot       | Bacterialblight   | 0           |
| bs5.png      | Brownspot       | Leafsmut          | 0           |
| ls1.png      | Leafsmut        | Brownspot         | 0           |
| ls2.png      | Leafsmut        | Brownspot         | 0           |
| ls3.png      | Leafsmut        | Brownspot         | 0           |
| ls4.png      | Leafsmut        | Leafsmut          | 1           |
| ls5.png      | Leafsmut        | Brownspot         | 0           |

Overall score: 4/15