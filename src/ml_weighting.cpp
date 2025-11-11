#include "axis/ml_weighting.h"
#include <chrono>
#include <iostream>

namespace axis {

MLWeightingEngine::MLWeightingEngine() = default;

bool MLWeightingEngine::loadModel(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << json_path << std::endl;
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();
    
    // Parse model type
    std::string model_type_str = extractJsonValue(json_content, "model_type");
    if (model_type_str == "GBDT") {
        model_type_ = ModelType::GBDT;
        return parseGBDTModel(json_content);
    } else if (model_type_str == "MLP") {
        model_type_ = ModelType::MLP;
        return parseMLPModel(json_content);
    } else {
        std::cerr << "Unknown model type: " << model_type_str << std::endl;
        return false;
    }
}

Eigen::VectorXd MLWeightingEngine::computeWeights(const SensorFeatures& features) {
    if (!model_loaded_) {
        // Return default weights if no model is loaded
        Eigen::VectorXd default_weights(5);
        default_weights << 1.0, 1.0, 1.0, 1.0, 1.0;
        return default_weights;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::VectorXd feature_vector = features.toVector();
    Eigen::VectorXd raw_weights;
    
    if (model_type_ == ModelType::GBDT) {
        raw_weights = predictGBDT(feature_vector);
    } else {
        raw_weights = predictMLP(feature_vector);
    }
    
    Eigen::VectorXd final_weights = normalizeAndClipWeights(raw_weights);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (duration.count() > 1000) { // Warning if inference takes >1ms
        std::cerr << "ML inference took " << duration.count() << " microseconds (target <1000μs)" << std::endl;
    }
    
    return final_weights;
}

void MLWeightingEngine::setWeightBounds(double min_weight, double max_weight) {
    if (min_weight > 0.0 && max_weight > min_weight) {
        min_weight_ = min_weight;
        max_weight_ = max_weight;
    } else {
        std::cerr << "Invalid weight bounds: min=" << min_weight << ", max=" << max_weight << std::endl;
    }
}

bool MLWeightingEngine::parseGBDTModel(const std::string& json_content) {
    try {
        // Parse learning rate
        std::string lr_str = extractJsonValue(json_content, "learning_rate");
        gbdt_learning_rate_ = std::stod(lr_str);
        
        // Parse initial prediction
        std::string init_str = extractJsonValue(json_content, "init_prediction");
        gbdt_init_prediction_ = std::stod(init_str);
        
        // Parse trees
        std::string trees_str = extractJsonValue(json_content, "trees");
        std::vector<std::string> tree_strs = splitJsonArray(trees_str);
        
        gbdt_trees_.clear();
        for (const auto& tree_str : tree_strs) {
            GBDTTree tree;
            tree.root = parseGBDTNode(tree_str);
            if (tree.root) {
                gbdt_trees_.push_back(std::move(tree));
            }
        }
        
        if (!gbdt_trees_.empty()) {
            model_loaded_ = true;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing GBDT model: " << e.what() << std::endl;
    }
    
    return false;
}

bool MLWeightingEngine::parseMLPModel(const std::string& json_content) {
    try {
        // Parse layers
        std::string layers_str = extractJsonValue(json_content, "layers");
        std::vector<std::string> layer_strs = splitJsonArray(layers_str);
        
        mlp_model_.layers.clear();
        for (const auto& layer_str : layer_strs) {
            MLPLayer layer;
            
            // Parse weights matrix
            std::string weights_str = extractJsonValue(layer_str, "weights");
            std::vector<std::string> weight_rows = splitJsonArray(weights_str);
            
            int rows = weight_rows.size();
            int cols = 0;
            if (!weight_rows.empty()) {
                std::vector<std::string> first_row = splitJsonArray(weight_rows[0]);
                cols = first_row.size();
            }
            
            layer.weights.resize(rows, cols);
            for (int i = 0; i < rows; ++i) {
                std::vector<std::string> row_values = splitJsonArray(weight_rows[i]);
                for (int j = 0; j < cols; ++j) {
                    layer.weights(i, j) = std::stod(row_values[j]);
                }
            }
            
            // Parse bias vector
            std::string bias_str = extractJsonValue(layer_str, "bias");
            std::vector<std::string> bias_values = splitJsonArray(bias_str);
            
            layer.bias.resize(rows);
            for (int i = 0; i < rows; ++i) {
                layer.bias[i] = std::stod(bias_values[i]);
            }
            
            // Parse activation function
            layer.activation = extractJsonValue(layer_str, "activation");
            if (layer.activation.empty()) {
                layer.activation = "relu";
            }
            
            mlp_model_.layers.push_back(std::move(layer));
        }
        
        // Parse output scaling
        std::string scale_str = extractJsonValue(json_content, "output_scale");
        if (!scale_str.empty()) {
            mlp_model_.output_scale = std::stod(scale_str);
        }
        
        std::string bias_str = extractJsonValue(json_content, "output_bias");
        if (!bias_str.empty()) {
            mlp_model_.output_bias = std::stod(bias_str);
        }
        
        if (!mlp_model_.layers.empty()) {
            model_loaded_ = true;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing MLP model: " << e.what() << std::endl;
    }
    
    return false;
}

std::unique_ptr<GBDTNode> MLWeightingEngine::parseGBDTNode(const std::string& node_json) {
    auto node = std::make_unique<GBDTNode>();
    
    std::string is_leaf_str = extractJsonValue(node_json, "is_leaf");
    if (is_leaf_str == "true") {
        node->is_leaf = true;
        std::string value_str = extractJsonValue(node_json, "value");
        node->leaf_value = std::stod(value_str);
        return node;
    }
    
    // Parse split node
    std::string feature_str = extractJsonValue(node_json, "feature_index");
    std::string threshold_str = extractJsonValue(node_json, "threshold");
    
    node->feature_index = std::stoi(feature_str);
    node->threshold = std::stod(threshold_str);
    node->is_leaf = false;
    
    // Parse children
    std::string left_str = extractJsonValue(node_json, "left_child");
    std::string right_str = extractJsonValue(node_json, "right_child");
    
    if (!left_str.empty()) {
        node->left_child = parseGBDTNode(left_str);
    }
    if (!right_str.empty()) {
        node->right_child = parseGBDTNode(right_str);
    }
    
    return node;
}

Eigen::VectorXd MLWeightingEngine::predictGBDT(const Eigen::VectorXd& features) {
    Eigen::VectorXd predictions(5);
    predictions.setConstant(gbdt_init_prediction_);
    
    // For each output dimension, accumulate tree predictions
    for (int output_dim = 0; output_dim < 5; ++output_dim) {
        double prediction = gbdt_init_prediction_;
        for (const auto& tree : gbdt_trees_) {
            prediction += gbdt_learning_rate_ * tree.predict(features);
        }
        predictions[output_dim] = prediction;
    }
    
    return predictions;
}

Eigen::VectorXd MLWeightingEngine::predictMLP(const Eigen::VectorXd& features) {
    return mlp_model_.predict(features);
}

Eigen::VectorXd MLWeightingEngine::normalizeAndClipWeights(const Eigen::VectorXd& raw_weights) {
    Eigen::VectorXd weights = raw_weights;
    
    // Clip to bounds
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = std::max(min_weight_, std::min(max_weight_, weights[i]));
    }
    
    // Normalize to maintain relative ratios
    double mean_weight = weights.mean();
    if (mean_weight > 0.0) {
        weights = weights / mean_weight; // Normalize to mean=1.0
    }
    
    return weights;
}

std::string MLWeightingEngine::extractJsonValue(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\":";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) {
        return "";
    }
    
    pos += search_key.length();
    
    // Skip whitespace
    while (pos < json.length() && std::isspace(json[pos])) {
        ++pos;
    }
    
    if (pos >= json.length()) {
        return "";
    }
    
    // Extract value
    size_t start = pos;
    size_t end = pos;
    
    if (json[start] == '"' || json[start] == '[' || json[start] == '{') {
        // String, array, or object value
        char delimiter = json[start];
        if (delimiter == '"') {
            end = json.find('"', start + 1);
            if (end != std::string::npos) {
                return json.substr(start + 1, end - start - 1);
            }
        } else {
            // Array or object - find matching closing bracket/brace
            int depth = 0;
            char closing_delimiter = (delimiter == '[') ? ']' : '}';
            
            for (size_t i = start; i < json.length(); ++i) {
                if (json[i] == delimiter) {
                    ++depth;
                } else if (json[i] == closing_delimiter) {
                    --depth;
                    if (depth == 0) {
                        end = i + 1;
                        break;
                    }
                }
            }
            
            if (end > start) {
                return json.substr(start, end - start);
            }
        }
    } else {
        // Numeric value
        while (end < json.length() && 
               json[end] != ',' && json[end] != '}' && json[end] != ']' && 
               !std::isspace(json[end])) {
            ++end;
        }
        
        return json.substr(start, end - start);
    }
    
    return "";
}

std::vector<std::string> MLWeightingEngine::splitJsonArray(const std::string& array_str) {
    std::vector<std::string> elements;
    
    if (array_str.empty() || array_str[0] != '[') {
        return elements;
    }
    
    std::string content = array_str.substr(1, array_str.length() - 2); // Remove [ ]
    
    size_t start = 0;
    int depth = 0;
    
    for (size_t i = 0; i < content.length(); ++i) {
        if (content[i] == '[' || content[i] == '{') {
            ++depth;
        } else if (content[i] == ']' || content[i] == '}') {
            --depth;
        } else if (content[i] == ',' && depth == 0) {
            std::string element = content.substr(start, i - start);
            if (!element.empty()) {
                elements.push_back(element);
            }
            start = i + 1;
        }
    }
    
    // Add last element
    if (start < content.length()) {
        std::string element = content.substr(start);
        if (!element.empty()) {
            elements.push_back(element);
        }
    }
    
    return elements;
}

} // namespace axis
