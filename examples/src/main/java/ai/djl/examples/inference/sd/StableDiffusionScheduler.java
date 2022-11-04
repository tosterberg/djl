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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

public class StableDiffusionScheduler {
    private final int numTrainTimesteps = 1000;
    private int numInferenceSteps;
    private float betaStart = (float) 0.00085;
    private float betaEnd = (float) 0.012;
    private NDManager manager;
    private NDArray betas;
    private NDArray alphas;
    private NDArray alphasCumProd;
    private float finalAlphaCumProd;
    private int counter = 0;
    private NDArray curSample = null;
    private NDList ets = new NDList();
    private int stepSize;
    public NDArray timesteps;

    private StableDiffusionScheduler() {}

    public StableDiffusionScheduler(NDManager mgr) {
        manager = mgr;
        betas = manager.linspace((float) Math.sqrt(betaStart),(float) Math.sqrt(betaEnd), numTrainTimesteps);
        betas = betas.mul(betas);
        alphas = manager.ones(betas.getShape()).add(betas.neg());
        alphasCumProd = manager.create(cumProd(alphas));
        finalAlphaCumProd = alphasCumProd.get(0).toFloatArray()[0];
    }

    private float[] cumProd(NDArray array) {
        float cumulative = 1;
        float[] alphasCumProdArr = new float[numTrainTimesteps];
        float[] alphasArr = array.toFloatArray();
        for (int i = 0; i < alphasCumProdArr.length; i++) {
            alphasCumProdArr[i] = alphasArr[i] * cumulative;
            cumulative = alphasCumProdArr[i];
        }
        return alphasCumProdArr;
    }

    public void setTimesteps(int inferenceSteps, int offset) {
        numInferenceSteps = inferenceSteps;
        stepSize = numTrainTimesteps / numInferenceSteps;
        timesteps = manager.arange(0, numInferenceSteps).mul(stepSize).add(offset);
    }
    public NDArray step(NDArray modelOutput, NDArray timestep, NDArray sample) {
        NDArray prevTimestep = manager.create(timestep.getInt() - stepSize);
        if (counter != 1) {
            ets.add(modelOutput);
        } else {
            prevTimestep = timestep.duplicate();
            timestep.add(-stepSize);
        }

        if (ets.size() == 1 && counter == 0) {
            curSample = sample;
        } else if(ets.size() == 1 && counter == 1) {
            modelOutput = modelOutput.add(ets.get(0)).div(2);
            sample = curSample;
            curSample = null;
        } else if(ets.size() == 2) {
            NDArray firstModel = ets.get(ets.size()-1).mul(3);
            NDArray secondModel = ets.get(ets.size()-2).mul(-1);
            modelOutput = firstModel.add(secondModel);
            modelOutput = modelOutput.div(2);
        } else if(ets.size() == 3) {
            NDArray firstModel = ets.get(ets.size()-1).mul(23);
            NDArray secondModel = ets.get(ets.size()-2).mul(-16);
            NDArray thirdModel = ets.get(ets.size()-3).mul(5);
            modelOutput = firstModel.add(secondModel).add(thirdModel);
            modelOutput = modelOutput.div(12);
        } else {
            NDArray firstModel = ets.get(ets.size()-1).mul(55);
            NDArray secondModel = ets.get(ets.size()-2).mul(-59);
            NDArray thirdModel = ets.get(ets.size()-3).mul(37);
            NDArray fourthModel = ets.get(ets.size()-4).mul(-9);
            modelOutput = firstModel.add(secondModel).add(thirdModel).add(fourthModel);
            modelOutput = modelOutput.div(24);
        }

        NDArray prevSample = getPrevSample(sample, timestep, prevTimestep, modelOutput);
        prevSample.setName("prev_sample");
        counter++;

        return prevSample;
    }

    private NDArray getPrevSample(NDArray sample, NDArray timestep, NDArray prevTimestep, NDArray modelOutput) {
        float alphaProdT = alphasCumProd.toFloatArray()[timestep.getInt()];
        float alphaProdTPrev;

        if (prevTimestep.getInt() >= 0) {
            alphaProdTPrev = alphasCumProd.toFloatArray()[prevTimestep.getInt()];
        } else {
            alphaProdTPrev = finalAlphaCumProd;
        }

        float betaProdT = 1 - alphaProdT;
        float betaProdTPrev = 1 - alphaProdTPrev;

        float sampleCoeff = (float) Math.sqrt(alphaProdTPrev / alphaProdT);
        float modelOutputCoeff = alphaProdT * (float) Math.sqrt(betaProdTPrev)
                + (float) Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev);

        sample = sample.mul(sampleCoeff);
        modelOutput = modelOutput.mul(alphaProdTPrev - alphaProdT);
        modelOutput = modelOutput.div(modelOutputCoeff);
        modelOutput = modelOutput.neg();
        return sample.add(modelOutput);
    }
}
