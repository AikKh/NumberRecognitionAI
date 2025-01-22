#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(double learning_rate, double decay_rate, const std::vector<int>& sizes, double(*active_func)(double), double(*active_fucn_der)(double))
    : m_initial_lr{ learning_rate }, m_lr{ learning_rate }, m_decay_rate{ decay_rate }, m_active_func{ active_func }, m_active_func_der{ active_fucn_der }
{
    for (int i = 1; i < sizes.size(); i++)
    {
        m_layers.push_back(Layer{ sizes[i - 1], sizes[i] });
    }
    m_layers.push_back(Layer{ sizes.back(), 0 });
}

const std::vector<double>& NeuralNetwork::Forward(const std::vector<double>& X)
{
    if (m_layers.front().Size != X.size())
    {
        throw std::invalid_argument("The dimentions do not match in the input");
    }

    std::copy(X.begin(), X.end(), m_layers.front().Neurons.begin());

    for (int i = 1; i < m_layers.size(); i++)
    {
        auto& layer1 = m_layers[i - 1];
        auto& layer2 = m_layers[i];

        for (int j = 0; j < layer2.Size; ++j)
        {
            layer2.Neurons[j] = 0;

            for (int k = 0; k < layer1.Size; ++k)
            {
                layer2.Neurons[j] += layer1.Neurons[k] * layer1.Weights[k][j];
            }
            layer2.Neurons[j] += layer2.Biases[j];
            layer2.Neurons[j] = (m_active_func)(layer2.Neurons[j]);
        }
    }

    return m_layers.back().Neurons;
}

void NeuralNetwork::Backpropagation(const std::vector<double>& y, bool accumulate)
{
    if (m_layers.back().Size != y.size())
    {
        throw std::invalid_argument("The dimentions do not match in the output");
    }

    // Compute output layer error
    std::vector<double> softmax_output = Softmax(m_layers.back().Neurons);
    std::vector<double> errors(m_layers.back().Size);

    for (int i = 0; i < m_layers.back().Size; i++)
    {
        errors[i] = y[i] - softmax_output[i];
    }

    for (int k = m_layers.size() - 2; k >= 0; k--)
    {
        Layer& layer1 = m_layers[k];
        Layer& layer2 = m_layers[k + 1];

        std::vector<double> gradients(layer2.Size);
        for (int i = 0; i < layer2.Size; i++)
        {
            gradients[i] = errors[i] * (m_active_func_der)(layer2.Neurons[i]);
        }

        // Update or accomulate the error
        for (int i = 0; i < layer2.Size; i++)
        {
            for (int j = 0; j < layer1.Size; j++)
            {
                if (accumulate) layer1.WeightGradients[j][i] += gradients[i] * layer1.Neurons[j];
                else layer1.Weights[j][i] += gradients[i] * layer1.Neurons[j] * m_lr;
            }

            if (accumulate) layer2.BiasGradients[i] += gradients[i];
            else layer2.Biases[i] += gradients[i] * m_lr;
        }

        // Calculate error for the next layer (backpropagate error)
        std::vector<double> errorsNext(layer1.Size, 0.0);
        for (int i = 0; i < layer1.Size; i++)
        {
            for (int j = 0; j < layer2.Size; j++)
            {
                errorsNext[i] += layer1.Weights[i][j] * errors[j];
            }
        }

        errors = errorsNext;
    }
}

void NeuralNetwork::UpdateWeights(int batch_size)
{
    for (auto& layer : m_layers)
    {
        layer.UpdateWeights(m_lr / static_cast<double>(batch_size));
    }
}
