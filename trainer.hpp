#pragma once

#include "neural_network.hpp"
#include "activation_functions.hpp"
#include "loader.hpp"

class Trainer {
public:
	Trainer(const std::string& folder_path,
		const std::string& training_images_filepath,
		const std::string& training_labels_filepath,
		const std::string& test_images_filepath,
		const std::string& test_labels_filepath) :
		m_nn{ 0.1, 0.003, { 784, 512, 128, 10 }, &ActivationFunctions::LeakyReLU, &ActivationFunctions::LeakyReLUDerivative },
		m_loader{ folder_path, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath }
	{
	}

	NeuralNetwork& Run()&
	{
		auto [train, test] = m_loader.LoadData();

		auto& [x_train, y_train] = train;
		auto& [x_test, y_test] = test;

		std::cout << "Loaded data" << std::endl;

		Train(NormalizeData(x_train), y_train);
		double accuracy = Test<false>(NormalizeData(x_test), y_test, 10000);

		std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;

		Test<true>(NormalizeData(x_test), y_test, 10);

		return m_nn;
	}

	void Train(const std::vector<std::vector<double>>& data, const std::vector<uint8_t>& labels)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> range(0, data.size());

		const int count = data.size();
		static constexpr size_t batch_size = 64 * 2;

		std::vector<double> target(10);
		int correct_predictions = 0;

		for (size_t epoch = 0; epoch < 300; ++epoch)
		{
			correct_predictions = 0;

			m_nn.InitializeGradientAccumulators();

			for (int j = 0; j < batch_size; j++)
			{
				int index = range(gen);

				std::fill(target.begin(), target.end(), 0);
				target[labels[index]] = 1.0;

				auto& output = m_nn.Forward(data[index]);

				int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

				if (labels[index] == predicted_label) correct_predictions++;

				m_nn.Backpropagation(target, true);
			}

			m_nn.UpdateWeights(batch_size);

			m_nn.UpdateLearningRate(epoch);

			std::cout << "Epoch: " << epoch << ", Accuracy: " << (correct_predictions / static_cast<double>(batch_size) * 100.0) << '%' << std::endl;
		}

		std::cout << "Last train Accuracy: " << Test<false>(data, labels, 10000) * 100.0 << "%" << std::endl;
	}

	template<bool debug>
	double Test(const std::vector<std::vector<double>>& data, const std::vector<uint8_t>& labels, size_t count)
	{
		count = std::min(data.size(), count);

		int correct_predictions = 0;
		for (int i = 0; i < count; ++i)
		{
			auto& output = m_nn.Forward(data[i]);

			int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

			if constexpr (debug)
			{
				Show(data[i], output);
				std::cout << "Predicted label: " << predicted_label << std::endl;
			}

			if (predicted_label == labels[i])
			{
				correct_predictions++;
			}
		}

		return static_cast<double>(correct_predictions) / static_cast<double>(count);
	}

	void Show(const std::vector<double>& image, const std::vector<double>& logits)
	{
		for (int i = 0; i < 28; ++i)
		{
			for (int j = 0; j < 28; ++j)
			{
				double c = image[i * 28 + j];
				std::cout << (c > 0?'#':' ');
				std::cout << ' ';
			}

			std::cout << std::endl;
		}

		auto probs = NeuralNetwork::Softmax(logits);

		for (int i = 0; i < probs.size(); i++)
		{
			std::cout << i << ": " << probs[i] * 100 << "%";
			std::cout << ',';
		}
		std::cout << std::endl;

		int predicted_label = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
		std::cout << "Predict: " << predicted_label << std::endl;
	}

private:
	std::vector<std::vector<double>> NormalizeData(const MnistDataloader::InputType& data)
	{
		std::vector<std::vector<double>> normalized_data;
		normalized_data.reserve(data.size());

		std::vector<double> normilized(data[0].size());

		for (auto& image : data)
		{
			std::transform(image.begin(), image.end(), normilized.begin(), [](uint8_t x) { return static_cast<double>(x) / 255.0; });
			normalized_data.push_back(normilized);
		}

		return normalized_data;
	}

private:
	NeuralNetwork m_nn;
	MnistDataloader m_loader;
};