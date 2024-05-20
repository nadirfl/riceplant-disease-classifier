package ch.zhaw.deeplearningjava.ferlinad.riceplantdiseases;


import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomFlipTopBottom;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

public final class Training {
    
    // number of training samples to process before updating the model
    private static final int BATCH_SIZE = 32;

    // number of passes over complete dataset
    private static final int EPOCHS = 2;

    public static void main(String[] args) throws IOException, TranslateException{
        // location of model
        Path modelDir = Paths.get("models");

        // create ImageFolder dataset from directory
        ImageFolder dataset = initDataset("rice_leafs");
        // Spliting into training and validation dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);
        Dataset training = datasets[0];
        Dataset validation = datasets[1];

        // set loss function, which seeks to minimize errors
        // loss function evaluates model's predictions against the correct answer (during training)
        // higher numbers are bad - means model performed poorly; indicates more errors;
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // hyperparameters for training
        TrainingConfig config = setupTrainingConfig(loss);

        //empty model to hold patterns
        Model model = Models.getModel();
        Trainer trainer = model.newTrainer(config);

        // collect metrics and report KPIs
        trainer.setMetrics(new Metrics());

        Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_WIDTH);

        // initialize trainer with input shape
        trainer.initialize(inputShape);

        // find patterns in the data
        EasyTrain.fit(trainer, EPOCHS, training, validation);

        // set model properties
        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        // save model after training for inference
        model.save(modelDir, Models.MODEL_NAME);

        // save labels into model directory
        Models.saveSynset(modelDir, dataset.getSynset());
    }

    private static ImageFolder initDataset(String datasetRoot) throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
            .setRepositoryPath(Paths.get(datasetRoot))
            .optMaxDepth(10) 
            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
            .addTransform(new RandomFlipLeftRight())
            .addTransform(new RandomFlipTopBottom())
            .addTransform(new ToTensor())
            .setSampling(BATCH_SIZE, true)
            .build();
        
        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
            .addEvaluator(new Accuracy())
            .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}

