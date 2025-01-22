#pragma once

#include <vector>
#include <algorithm>
#include <exception>

#include "layer.hpp"
#include "activation_functions.hpp"

class NeuralNetwork {
public:
	NeuralNetwork(double learning_rate, double decay_rate, const std::vector<int>& sizes, double(*active_func)(double), double(*active_fucn_der)(double));

	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork(NeuralNetwork&&) = delete;

	const std::vector<double>& Forward(const std::vector<double>& X);

	void Backpropagation(const std::vector<double>& y, bool accumulate = false);

	void UpdateWeights(int batch_size);

	void UpdateLearningRate(int epoch)
	{
		m_lr = m_initial_lr / (1.0 + m_decay_rate * epoch);
	}

	void InitializeGradientAccumulators()
	{
		for (auto& layer : m_layers)
		{
			layer.InitializeGradientAccumulators();
		}
	}

	static std::vector<double> Softmax(const std::vector<double>& z)
	{
		std::vector<double> softmax(z.size());
		double max_val = *std::max_element(z.begin(), z.end()); // Stability improvement
		double sum = 0.0;

		for (const double val:z)
		{
			sum += std::exp(val - max_val);
		}

		for (size_t i = 0; i < z.size(); i++)
		{
			softmax[i] = std::exp(z[i] - max_val) / sum;
		}

		return softmax;
	}


private:
	double m_initial_lr;
	double m_lr;
	double m_decay_rate;

	std::vector<Layer> m_layers;

	double (*m_active_func)(double);
	double (*m_active_func_der)(double);
};