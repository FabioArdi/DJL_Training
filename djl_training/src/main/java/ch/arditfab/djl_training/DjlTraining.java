package ch.arditfab.djl_training;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import ai.djl.translate.TranslateException;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;

public class DjlTraining {

    public static final String MODEL_NAME = "shoeclassifier";
    public static final int IMAGE_WIDTH = 224;
    public static final int IMAGE_HEIGHT = 224;
    public static final int EPOCHS = 2;
    public static final int BATCH_SIZE = 20;
    public static final int NUM_OF_OUTPUTS = 40;

    public static void main(String[] args) throws IOException, TranslateException {
    
        ImageFolder dataset = initDataset("djl_training/src/main/resources/static/data/ut-zap50k-images-square");
        RandomAccessDataset[] datasets = dataset.randomSplit(8,2);

        Loss loss = Loss.softmaxCrossEntropyLoss();
        TrainingConfig config = setupTrainingConfig(loss);

        try (Model model = getModel();
                Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, 3, IMAGE_HEIGHT, IMAGE_HEIGHT);
            trainer.initialize(inputShape);
            EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(EPOCHS));
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            model.save(Path.of("src/main/resources"), MODEL_NAME);
        }

    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .setSampling(BATCH_SIZE, true)
                .build();
        dataset.prepare();
        return dataset;
    }

    public static Model getModel() {
        Model model = Model.newInstance(MODEL_NAME);
        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
                .setNumLayers(50)
                .setOutSize(NUM_OF_OUTPUTS)
                .build();
        model.setBlock(resNet50);
        return model;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }


}
