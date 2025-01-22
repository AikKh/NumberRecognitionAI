#pragma once

#include <cmath>
#include <utility>

namespace ActivationFunctions
{
	static double Sigmoid(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	static double SigmoidDerivative(double x)
	{
		double sig = Sigmoid(x);
		return sig * (1 - sig);
	}

	static double ReLU(double x)
	{
		return std::max(0.0, x);
	}

	static double ReLUDerivative(double x)
	{
		return x > 0?1.0:0.0;
	}

	static inline double Tanh(double x)
	{
		return std::tanh(x);
	}

	static inline double TanhDerivative(double x)
	{
		double tanh_x = Tanh(x);
		return 1.0 - tanh_x * tanh_x;
	}

	static inline double NormalizedTanh(double x)
	{
		return (std::tanh(x) + 1.0) / 2.0;
	}

	static inline double NormalizedTanhDerivative(double x)
	{
		double tanh_x = std::tanh(x);
		return (1.0 - tanh_x * tanh_x) / 2.0;
	}

	static double LeakyReLU(double x)
	{
		return (x > 0)?x:0.01 * x;
	}

	static double LeakyReLUDerivative(double x)
	{
		return (x > 0)?1.0:0.01;
	}
}