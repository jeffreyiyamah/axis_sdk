#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <fstream>
#include <chrono>
#include "axis/ml_weighting.h"

using axis::MLWeightingEngine;
using axis::SensorFeatures;

static std::string createTempModelFile()
{
    const char* path = "ml_test_model.json";
    std::ofstream ofs(path);
    ofs << "{\n"
        << "  \"model_type\": \"MLP\",\n"
        << "  \"layers\": [\n"
        << "    {\"weights\": [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], \"bias\": [0], \"activation\": \"linear\"}\n"
        << "  ],\n"
        << "  \"output_scale\": 1.0,\n"
        << "  \"output_bias\": 1.0\n"
        << "}\n";
    ofs.close();
    return std::string(path);
}

TEST(MLWeightingTest, ModelLoadingFromJSON)
{
    MLWeightingEngine engine;
    std::string path = createTempModelFile();
    ASSERT_TRUE(engine.loadModel(path));
    EXPECT_TRUE(engine.isModelLoaded());
    EXPECT_EQ(engine.getModelType(), MLWeightingEngine::ModelType::MLP);
}

TEST(MLWeightingTest, InferenceWithKnownInputs)
{
    MLWeightingEngine engine;
    std::string path = createTempModelFile();
    ASSERT_TRUE(engine.loadModel(path));

    SensorFeatures features;
    features.imu_noise_variance = 2.0;

    Eigen::VectorXd w = engine.computeWeights(features);

    ASSERT_EQ(w.size(), 5);
    for (int i = 0; i < w.size(); ++i) {
        EXPECT_NEAR(w[i], 1.0, 1e-6);
    }
}

TEST(MLWeightingTest, WeightsWithinBounds)
{
    MLWeightingEngine engine;
    std::string path = createTempModelFile();
    ASSERT_TRUE(engine.loadModel(path));

    engine.setWeightBounds(0.5, 1.5);
    auto bounds = engine.getWeightBounds();
    EXPECT_DOUBLE_EQ(bounds.first, 0.5);
    EXPECT_DOUBLE_EQ(bounds.second, 1.5);

    SensorFeatures features;
    features.imu_noise_variance = 1000.0;

    Eigen::VectorXd w = engine.computeWeights(features);
    ASSERT_EQ(w.size(), 5);
    for (int i = 0; i < w.size(); ++i) {
        EXPECT_GE(w[i], 0.5);
        EXPECT_LE(w[i], 1.5);
    }
}

TEST(MLWeightingTest, RealTimePerformance)
{
    MLWeightingEngine engine;
    std::string path = createTempModelFile();
    ASSERT_TRUE(engine.loadModel(path));

    SensorFeatures features;

    const int N = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        (void)engine.computeWeights(features);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double avg_us = static_cast<double>(us) / N;
    EXPECT_LT(avg_us, 1000.0);
}
