/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.examples.inference.sd;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.*;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/** The example is targeted to a specific use case of Stable Diffusion. */
public class StableDiffusion {
    private static final Logger logger = LoggerFactory.getLogger(StableDiffusion.class);
    private static final String sdArtifacts =
            "/absolute/path/to/stable-diffusion/artifacts";
    private static final String prompt = "an astronaut riding a horse";
    private static final int hieght = 512;
    private static final int width = 512;
    private static final int UNKNOWN_TOKEN = 49407;
    private static final int MAX_LENGTH = 77;
    private static final int steps = 50;
    private static final int offset = 1;
    private static final float guidanceScale = (float) 7.5;
    private static final Engine engine = Engine.getEngine("PyTorch");
    private static final NDManager manager =
            NDManager.newBaseManager(engine.defaultDevice(), engine.getEngineName());
    private static final HuggingFaceTokenizer tokenizer;

    static {
        try {
            tokenizer =
                    HuggingFaceTokenizer.builder()
                            .optManager(manager)
                            .optPadding(true)
                            .optPadToMaxLength()
                            .optMaxLength(MAX_LENGTH)
                            .optTruncation(true)
                            .optTokenizerName("sentence-transformers/msmarco-distilbert-dot-v5")
                            .build();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private StableDiffusion() {
    }

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        NDList textEncoding = SDTextEncoder(SDTextTokenizer(prompt));
        NDList uncondEncoding = SDTextEncoder(SDTextTokenizer(""));
        NDArray embeddings = textEncoding.get(1).concat(uncondEncoding.get(1));

        Shape shape = new Shape(1, 4, hieght / 8, width / 8);
        NDArray latent = manager.randomNormal(shape);

        StableDiffusionScheduler scheduler = new StableDiffusionScheduler(manager);
        scheduler.setTimesteps(steps, offset);
        Predictor<NDList, NDList> predictor = SDUNetPredictor();

        for (int i = (int) scheduler.timesteps.size() - 1; i >= 0; i--) {
            NDArray t = manager.create(scheduler.timesteps.toArray()[i]);
            NDArray latentModelOutput = latent.concat(latent);
            NDArray noisePred = predictor.predict(buildUnetInput(embeddings, t, latentModelOutput)).get(0);

            NDList splitNoisePred = noisePred.split(2);
            NDArray noisePredText = splitNoisePred.get(0);
            NDArray noisePredUncond = splitNoisePred.get(1);

            NDArray scaledNoisePredUncond = noisePredText.add(noisePredUncond.neg());
            scaledNoisePredUncond = scaledNoisePredUncond.mul(guidanceScale);
            noisePred = noisePredUncond.add(scaledNoisePredUncond);

            latent = scheduler.step(noisePred, t, latent);
        }
        saveImage(latent);

        logger.info("Stable diffusion image generated from prompt: \"{}\".", prompt);
    }

    private static void saveImage(NDArray input) throws TranslateException, ModelNotFoundException,
            MalformedModelException, IOException {
        input = input.div(0.18215);

        NDList encoded = new NDList();
        encoded.add(input);

        NDList z = SDPostQuantConv(encoded);
        NDList decoded = SDDecoder(z);
        NDArray scaled = decoded.get(0).div(2).add(0.5).clip(0, 1);

        scaled = scaled.transpose(0, 2, 3, 1);
        scaled = scaled.mul(255).round().toType(DataType.INT8, true).get(0);
        Image image = BufferedImageFactory.getInstance().fromNDArray(scaled);

        saveImage(image, "out", "repo/path/to/output/location");
    }

    private static NDList buildUnetInput(NDArray input, NDArray timestep, NDArray latents) {
        input.setName("encoder_hidden_states");
        NDList list = new NDList();
        list.add(latents);
        list.add(timestep);
        list.add(input);
        return list;
    }

    private static NDList SDTextEncoder(NDList input)
            throws ModelNotFoundException, MalformedModelException, IOException,
            TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(sdArtifacts)
                        .optModelName("traced_text_encoder_model")
                        .optEngine(engine.getEngineName())
                        .optProgress(new ProgressBar())
                        .optTranslator(new NoopTranslator())
                        .build();

        ZooModel<NDList, NDList> model = criteria.loadModel();
        Predictor<NDList, NDList> predictor = model.newPredictor();
        NDList output = predictor.predict(input);
        model.close();
        return output;
    }

    private static Predictor<NDList, NDList> SDUNetPredictor()
            throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(sdArtifacts)
                        .optModelName("traced_unet_model")
                        .optEngine(engine.getEngineName())
                        .optProgress(new ProgressBar())
                        .optTranslator(new NoopTranslator())
                        .build();

        ZooModel<NDList, NDList> model = criteria.loadModel();
        return model.newPredictor();
    }

    private static NDList SDPostQuantConv(NDList input)
            throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(sdArtifacts)
                        .optModelName("traced_post_quant_conv")
                        .optEngine(engine.getEngineName())
                        .optTranslator(new NoopTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        ZooModel<NDList, NDList> model = criteria.loadModel();
        Predictor<NDList, NDList> predictor = model.newPredictor();
        NDList output = predictor.predict(input);
        model.close();
        return output;
    }

    private static NDList SDDecoder(NDList input)
            throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(sdArtifacts)
                        .optModelName("traced_decoder")
                        .optEngine(engine.getEngineName())
                        .optTranslator(new NoopTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        ZooModel<NDList, NDList> model = criteria.loadModel();
        Predictor<NDList, NDList> predictor = model.newPredictor();
        NDList output = predictor.predict(input);
        predictor.close();
        return output;
    }

    public static void saveImage(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }


    private static NDList SDTextTokenizer(String prompt) {
        List<String> tokens = tokenizer.tokenize(prompt);
        int[][] tokenValues = new int[1][MAX_LENGTH];
        ObjectMapper mapper = new ObjectMapper();
        File fileObj = new File(sdArtifacts + "/vocab_dictionary.json");
        try {
            Map<String, Integer> mapObj =
                    mapper.readValue(fileObj, new TypeReference<Map<String, Integer>>() {
                    });
            int counter = 0;
            for (String token : tokens) {
                if (mapObj.get(token) != null) {
                    tokenValues[0][counter] = mapObj.get(token);
                } else {
                    tokenValues[0][counter] = UNKNOWN_TOKEN;
                }
                counter++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        NDArray ndArray = manager.create(tokenValues);
        return new NDList(ndArray);
    }
}