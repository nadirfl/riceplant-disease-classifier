# Rice Leaf Image Classifier
## Quicklinks:
 - Run the application locally: ```RiceplantdiseasesApplication.main()```
 - Creating and training the model: ```Training.main()```

## Table of contents
1. Project goal and motivation
2. Data collection and description
3. Prerequisites
4. Project structure
5. Modeling and Training
6. Interpretation
7. Validation
8. Conclusion

## 1. Project goal and motivation

For this project, I aimed to build an application that addresses a real-world problem. Additionally, I was eager to delve deeper into the fields of Computer Vision and Image Classification, as these are captivating areas of study that hold significant potential for practical applications and future research.

After conducting research on existing datasets available on Kaggle, I discovered a dataset specifically for rice plant diseases. Upon reviewing the images and accompanying documentation, I decided to build a model using this dataset, as it presents a practical solution to a significant real-world problem in agriculture.

## 2. Data collection and description
Link to the dataset: https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset

*The Rice Life Disease Dataset is an extensive collection of data focused on three major diseases that affect rice plants: Bacterial Blight (BB), Brown Spot (BS), and Leaf Smut (LS).*

The dataset consists of 3 directories (as of 20th of May 2024):

    rice-plant-diseases-dataset
    ├── Bacterialblight: 1604 images
    ├── Brownspot:       1620 images
    └── Leafsmut:        1460 images

The images in the dataset are of decent quality, with each disease clearly visible and recognizable, making them suitable for training an image classification model.

Definition of the rice plant diseases:
- Bacterial blight: deadly bacterial disease that is among the most destructive afflictions of cultivated rice. In severe epidemics, crop loss may be as high as 75 percent, and millions of hectares of rice are infected annually
- Brown spot: a fungal disease that infects the coleoptile, leaves, leaf sheath, panicle branches, glumes, and spikelets. Its most observable damage is the numerous big spots on the leaves which can kill the whole leaf
- Leaf smut: a widely distributed, but somewhat minor, disease of rice. The fungus produces slightly raised, angular, black spots (sori) on both sides of the leaves. Although rare, it also can produce spots on leaf sheaths

## 3. Prerequisites
There are no specific requirements to run this application beyond the standard setup for Java 21, which we have already covered in the "Model Deployment and Maintenance" course.

However, to train the model locally, you will need to download the dataset, which you available [here](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset). I've downloaded the dataset May 20, 2024.

After downloading and unzipping the dataset, make sure to rename the folder to "rice_leafs" and place it in the root directory of the application (see the "Project structure" section for the exact location).

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
    │   |   │   └── styles.css                      # File containing the design for the UI
    │   |   └── application.properties              # File containing properties for the application
    |   
    ├── test_dataset                                # Directory consistinc of 15 images for model testing
    |
    ├── .gitignore                                  # File containing a list of excluded files/directories not to be pushed to GitHub
    ├── pom.xml                                     # File containing the dependencies
    └── README.md                                   # Project overview and documentation

## 5. Modeling and Training
For modeling I am using the [DeepJavaLibrary](https://djl.ai/). This library allows me to quickly build and fine-tune models and provides a variety of neural networks suitable for different types of problems.

I am using the Residual Network (ResNet), a generic implementation adapted from the original source by Wei Wu at https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py, and further refined by Antti-Pekka Hynninen. This implementation follows the original ResNet architecture, which won the ILSVRC 2015 competition, developed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Their groundbreaking work is titled "Deep Residual Learning for Image Recognition."

In the class ```Models.java```, I start off by building the first block of the network with the following code:
```java
Block resNet50 = ResNetV1.builder()
    .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
    .setNumLayers(NUM_OF_LAYERS)
    .setOutSize(NUM_OF_OUTPUT)
    .build();
```
The neural network consists of 50 layers and has 3 output layers (corresponding to the disease labels). The constants (e.g., IMAGE_HEIGHT) are defined within the class and have been adjusted for different runs (see details below).

The training is in the  ```Training.java``` class, where I use the dataset to train my model. To initiate the training process, you can start the ```main()``` method.

After experimenting with various settings, I have selected the following (hyper-)parameters (see test runs below for details):
- Training split: 80/20
- Training configuration: 
  - Loss function: Softmax Cross Entropy
  - Learning Rate Adjustment: MultiFactorTracker to reduce the learning rate after a certain number of epochs
  - Gradient Descent Optimizer
  - L2-regulation (weight decay of 0.0001)
  - Evaluator: Accuracy
- Image transformations
  - Resizing to 100x100
  - Random flip (left/right & top/bottom)
- Batch size before updating model: 32
- Epochs: 2

After training, the model (`riceLeafClassifier-0002.params`) and the labels (`synset.txt`) are saved in the directory `/models`.

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
- Performance a bit worse (0.82 vs 0.89 accuracy). 
- Keeping the transformations

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

### 5.5 Fifth Run
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
With a validation accuracy of 0.93 this is the best result yet. I will keep the current model for now.

### 5.6 Last Run
#### Changes 
Training Config
 - Add gradient descent and L2-regulation (weight decay of 0.0001)

#### Results
```
16:43:21.359 [main] WARN ai.djl.mxnet.jna.LibUtils -- No matching cuda flavor for win found: cu065mkl/sm_75.
16:43:21.572 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Training on: cpu().
16:43:21.574 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Load MXNet Engine Version 1.9.0 in 0.170 ms.
Training:    100% |========================================| Accuracy: 0.49, SoftmaxCrossEntropyLoss: 1.03
Validating:  100% |========================================|
16:48:27.676 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 1 finished.
16:48:27.677 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.50, SoftmaxCrossEntropyLoss: 1.03
16:48:27.677 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.41, SoftmaxCrossEntropyLoss: 1.42
Training:    100% |========================================| Accuracy: 0.56, SoftmaxCrossEntropyLoss: 0.94
Validating:  100% |========================================|
16:53:40.678 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Epoch 2 finished.
16:53:40.679 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Train: Accuracy: 0.56, SoftmaxCrossEntropyLoss: 0.94
16:53:40.679 [main] INFO ai.djl.training.listener.LoggingTrainingListener -- Validate: Accuracy: 0.61, SoftmaxCrossEntropyLoss: 0.89
```
#### Conclusion
Although the model performs worse in the validation phase, the out-of-sample tests (as discussed in the Validation chapter) look more promising. This new model appears to generalize better to unseen data.

## 6. Interpretation
After experimenting with different parameters and evaluating the performances, I've decided to use the latest model since it yielded the best performance in testing.

Here you can find an overview of the different test runs:

| Run nr. | Testing Accuracy | Testing Loss | Validation Accuracy | Validation Loss | Ellapsed training time |
| ------- | ---------------- | ------------ | ------------------- | --------------- | ---------------------- |
| 1       | 0.85             | 0.54         | 0.89                | 0.52            | 11min                  |
| 2       | 0.89             | 0.39         | 0.82                | 0.58            | 11min                  |
| 3       | 0.83             | 0.47         | 0.73                | 0.60            | **3h 30min**           |
| 4       | 0.77             | 0.60         | 0.79                | 0.55            | 13min                  |
| 5       | 0.79             | 0.75         | 0.93                | 0.27            | 10min                  |
| 6       | 0.56             | 0.94         | 0.61                | 0.89            | 10min                  |

## 7. Validation
For validating my model, I've created the directory `/test_dataset` consisting of 5 images per rice plant disease that I've manually downloaded from Google Images to preserve honest assessment. 

The distinct features per disease are the following (based on the dataset):
 - Bacterialblight: A lighter stripe following the leaf
 - Brownspot: brown spots on the leaf (as the name suggests)
 - Leafsmut: lighter spots with a dark border on the leaf

I decided to use a mix of easily distinguishable images and others that are more difficult to recognize in order to push the classification model to its limits.

Each prediction from my model gets a score of either 0 (incorrect) or 1 (correct) based on the prediction with the highest probability. I've first evaluated the simpler model from the fifth run:

| Picture Name | Actual Disease  | Predicted Disease | Score (0-1) |
| ------------ | --------------  | ----------------- | ----------- |
| bb1.png      | Bacterialblight | Bacterialblight   | 1           |
| bb2.png      | Bacterialblight | Bacterialblight   | 1           |
| bb3.png      | Bacterialblight | Leafsmut          | 0           |
| bb4.png      | Bacterialblight | Bacterialblight   | 1           |
| bb5.png      | Bacterialblight | Brownspot         | 0           |
| bs1.png      | Brownspot       | Bacterialblight   | 0           |
| bs2.png      | Brownspot       | Bacterialblight   | 0           |
| bs3.png      | Brownspot       | Bacterialblight   | 0           |
| bs4.png      | Brownspot       | Bacterialblight   | 0           |
| bs5.png      | Brownspot       | Bacterialblight   | 0           |
| ls1.png      | Leafsmut        | Leafsmut          | 1           |
| ls2.png      | Leafsmut        | Leafsmut          | 1           |
| ls3.png      | Leafsmut        | Bacterialblight   | 0           |
| ls4.png      | Leafsmut        | Brownspot         | 0           |
| ls5.png      | Leafsmut        | Bacterialblight   | 0           |

Overall score: 5/15

Noticeable problems: 
 - The prediction is highly dependent on the picture. 
 - In the picture *bb3.png* for example, there is a slight darker border around the affected area that might suggests a disease of leaf smut.
 - The same goes for the picture *bb5.png*, where darker parts might be (wrongfully) interpreted as brown spots
 - In picture *bs4.png*, there is an area on the leaf which reflects light and which might be (wrongfully) interpreted
 - In picture *ls5.png*, I would have expected a perfect prediction for Leafsmut since the diseases is clearly visible but the white spots are a bit longer than in the other pictures

Conclusion:
 - It appears that my model is overfitted.
 - The following attempts have been made:
   - Reduce Epoch to 1: maybe the insufficient amount of data is not enough for two epochs:
     - Validate: Accuracy: 0.92, SoftmaxCrossEntropyLoss: 0.23, Test score 3/15 -> even worse
   - Add gradient descent and L2-regulation (weight decay of 0.0001)
     - Validate: Accuracy: 0.61, SoftmaxCrossEntropyLoss: 0.89, Test score 8/15

After incorporating gradient descent and L2 regularization, the model seems to generalize better. Therefore, I have decided to keep the latest model. To further improve its performance, I would need a larger dataset.

| Picture Name | Actual Disease  | Predicted Disease | Score (0-1) |
| ------------ | --------------  | ----------------- | ----------- |
| bb1.png      | Bacterialblight | Bacterialblight   | 1           |
| bb2.png      | Bacterialblight | Bacterialblight   | 1           |
| bb3.png      | Bacterialblight | Bacterialblight   | 1           |
| bb4.png      | Bacterialblight | Bacterialblight   | 1           |
| bb5.png      | Bacterialblight | Bacterialblight   | 1           |
| bs1.png      | Brownspot       | Bacterialblight   | 0           |
| bs2.png      | Brownspot       | Bacterialblight   | 0           |
| bs3.png      | Brownspot       | Brownspot         | 1           |
| bs4.png      | Brownspot       | Leafsmut          | 0           |
| bs5.png      | Brownspot       | Bacterialblight   | 0           |
| ls1.png      | Leafsmut        | Bacterialblight   | 0           |
| ls2.png      | Leafsmut        | Bacterialblight   | 0           |
| ls3.png      | Leafsmut        | Leafsmut          | 1           |
| ls4.png      | Leafsmut        | Leafsmut          | 1           |
| ls5.png      | Leafsmut        | Bacterialblight   | 0           |

Overall score: 8/15

## 8. Conclusion
Throughout this project, I aimed to develop a practical solution to a real-world problem by creating a rice leaf disease classifier using deep learning techniques. Leveraging the Deep Java Library (DJL) and a dataset of rice plant diseases from Kaggle, I experimented with various configurations and transformations to optimize model performance.

Despite initial challenges with overfitting and varying results across different runs, the incorporation of gradient descent and L2 regularization improved the model's ability to generalize. The final model achieved a validation accuracy of 0.61 and demonstrated better generalization in out-of-sample tests, achieving an overall test score of 8/15.

These results indicate that while the current model shows promise, there is still room for improvement. To further enhance performance, a larger and more diverse dataset would be beneficial. Additionally, experimenting with advanced augmentation techniques and fine-tuning hyperparameters could lead to more robust and accurate predictions.

In conclusion, this project provided valuable insights into the complexities of training deep learning models for image classification and highlighted the importance of data quality and regularization techniques in achieving reliable results.