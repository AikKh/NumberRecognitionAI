#pragma once

#include <vector>
#include <random>
//#include "matrix.hpp"

struct Layer {
public:
	Layer(int size, int out_size) :
		Size{ size }, Neurons(size), Biases(size, 0.01), Weights(size, std::vector<double>(out_size)),
		WeightGradients(size, std::vector<double>(out_size)), BiasGradients(size)

	{
		double init_range = std::sqrt(2.0 / (size + out_size));

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(-init_range, init_range);

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < out_size; j++)
			{
				Weights[i][j] = dist(gen);
			}
		}
	}

	void InitializeGradientAccumulators()
	{
		for (auto& i : WeightGradients)
		{
			std::fill(i.begin(), i.end(), 0);
		}

		std::fill(BiasGradients.begin(), BiasGradients.end(), 0);
	}

	void UpdateWeights(double lr)
	{
		for (size_t i = 0; i < Weights.size(); i++)
		{
			for (size_t j = 0; j < Weights[i].size(); j++)
			{
				Weights[i][j] += lr * WeightGradients[i][j];
			}
		}
		for (size_t i = 0; i < Biases.size(); i++)
		{
			Biases[i] += lr * BiasGradients[i];
		}
	}

	int Size;
	std::vector<double> Neurons;
	std::vector<std::vector<double>> Weights;
	std::vector<double> Biases;

	std::vector<std::vector<double>> WeightGradients;
	std::vector<double> BiasGradients;
};