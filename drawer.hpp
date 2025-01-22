#pragma once

#include <SFML/Graphics.hpp>
#include "neural_network.hpp"

constexpr size_t RESOLUTION = 28;

class Drawer {
public:

	Drawer(int cell_size = 20) : m_size{ cell_size },
		m_data(RESOLUTION, std::vector<uint8_t>(RESOLUTION, 0)),
		m_window{ sf::VideoMode(RESOLUTION * cell_size /*+ 10 * cell_size*/, RESOLUTION * cell_size), "Drawer" }
	{
		m_window.setFramerateLimit(30);
	}

	void Run(NeuralNetwork& nn)
	{
		while (m_window.isOpen())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
				{
					m_window.close();
				}

				if (event.type == sf::Event::MouseButtonPressed)
				{
					if (event.mouseButton.button == sf::Mouse::Left)
					{
						m_left_pressed = true;
					}
					else if (event.mouseButton.button == sf::Mouse::Right)
					{
						m_right_pressed = true;
					}
				}

				if (event.type == sf::Event::MouseButtonReleased)
				{
					if (event.mouseButton.button == sf::Mouse::Left)
					{
						m_left_pressed = false;
					}
					else if (event.mouseButton.button == sf::Mouse::Right)
					{
						m_right_pressed = false;
					}
				}

				if (event.type == sf::Event::KeyPressed)
				{
					if (event.key.code == sf::Keyboard::C)
					{
						Clear();
					}
				}
			}

			if (m_left_pressed)
			{
				Predict(nn);
				FillCell(50);
			}
			else if (m_right_pressed)
			{
				FillCell(-100);
			}

			Draw();
		}
	}

private:
	void Predict(NeuralNetwork& nn) const
	{
		auto& logits = nn.Forward(NormalizeData());
		auto probs = NeuralNetwork::Softmax(logits);

		for (int i = 0; i < probs.size(); i++)
		{
			std::cout << i << ": [Prob:" << probs[i] * 100 << "%, Logits: " << logits[i] << "], ";
			if (i % 3 == 0) std::cout << std::endl;
		}
		std::cout << std::endl;

		int predicted_label = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
		std::cout << "Predict: " << predicted_label << std::endl;
	}

	std::vector<double> NormalizeData() const
	{
		std::vector<double> normalized;
		normalized.reserve(RESOLUTION * RESOLUTION);

		for (auto& row : m_data)
		{
			std::transform(row.begin(), row.end(), std::back_inserter(normalized), [](uint8_t x) { return static_cast<double>(x) / 255.0; });
		}

		return normalized;
	}

	void Draw()
	{
		m_window.clear();

		sf::RectangleShape cell{ sf::Vector2f(m_size, m_size) };

		for (int i = 0; i < RESOLUTION; ++i)
		{
			for (int j = 0; j < RESOLUTION; ++j)
			{
				uint8_t c = m_data[j][i];
				cell.setPosition(sf::Vector2f(i * m_size, j * m_size));
				cell.setFillColor({ c, c, c });

				m_window.draw(cell);
			}
		}

		m_window.display();
	}

	void Clear()
	{
		for (int i = 0; i < RESOLUTION; ++i)
		{
			for (int j = 0; j < RESOLUTION; ++j)
			{
				m_data[j][i] = 0;
			}
		}
	}

	void FillCell(int amount)
	{
		auto pos = sf::Mouse::getPosition(m_window);

		int x = pos.x / m_size;
		int y = pos.y / m_size;

		if (x < 0 || x >= RESOLUTION || y < 0 || y > RESOLUTION) return;

		m_data[y][x] = FitInRange(m_data[y][x] + amount);

		if (x + 1 < 28) m_data[y][x + 1] = FitInRange(m_data[y][x + 1] + amount);
		if (x - 1 >= 0) m_data[y][x - 1] = FitInRange(m_data[y][x - 1] + amount);
		if (y + 1 < 28) m_data[y + 1][x] = FitInRange(m_data[y + 1][x] + amount);
		if (y - 1 >= 0) m_data[y - 1][x] = FitInRange(m_data[y - 1][x] + amount);
	}

	inline static int FitInRange(int n)
	{
		return std::min(std::max(n, 0), 255);
	}

private:
	bool m_left_pressed = false;
	bool m_right_pressed = false;
	int m_size;
	std::vector<std::vector<uint8_t>> m_data;
	sf::RenderWindow m_window;
};