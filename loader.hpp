#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <string>

class MnistDataloader {
public:
    using InputType = std::vector<std::vector<uint8_t>>;
    using OutputType = std::vector<uint8_t>;

    MnistDataloader(const std::string& folder_path,
        const std::string& training_images_filepath,
        const std::string& training_labels_filepath,
        const std::string& test_images_filepath,
        const std::string& test_labels_filepath)
        : _folder_path(folder_path),
        _training_images_filepath(training_images_filepath),
        _training_labels_filepath(training_labels_filepath),
        _test_images_filepath(test_images_filepath),
        _test_labels_filepath(test_labels_filepath)
    {
    }

    std::pair<std::pair<InputType, OutputType>, std::pair<InputType, OutputType>> LoadData()
    {
        auto train_data = ReadImagesLabels(_folder_path + _training_images_filepath, _folder_path + _training_labels_filepath);
        auto test_data = ReadImagesLabels(_folder_path + _test_images_filepath, _folder_path + _test_labels_filepath);
        return { train_data, test_data };
    }

private:
    static std::pair<InputType, OutputType> ReadImagesLabels(const std::string& images_filepath, const std::string& labels_filepath)
    {
        // Read labels
        std::vector<uint8_t> labels;
        std::ifstream label_file(labels_filepath, std::ios::binary);
        if (!label_file)
        {
            throw std::runtime_error("Failed to open labels file: " + labels_filepath);
        }

        uint32_t magic = 0;
        uint32_t size = 0;
        label_file.read(reinterpret_cast<char*>(&magic), 4);
        label_file.read(reinterpret_cast<char*>(&size), 4);

        magic = swap_endian(magic);
        size = swap_endian(size);

        if (magic != 2049)
        {
            throw std::runtime_error("Magic number mismatch for labels file. Expected 2049, got " + std::to_string(magic));
        }

        labels.resize(size);
        label_file.read(reinterpret_cast<char*>(labels.data()), size);

        // Read images
        std::ifstream image_file(images_filepath, std::ios::binary);
        if (!image_file)
        {
            throw std::runtime_error("Failed to open images file: " + images_filepath);
        }

        uint32_t rows = 0;
        uint32_t cols = 0;

        image_file.read(reinterpret_cast<char*>(&magic), 4);
        image_file.read(reinterpret_cast<char*>(&size), 4);
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        image_file.read(reinterpret_cast<char*>(&cols), 4);

        magic = swap_endian(magic);
        size = swap_endian(size);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (magic != 2051)
        {
            throw std::runtime_error("Magic number mismatch for images file. Expected 2051, got " + std::to_string(magic));
        }

        std::vector<std::vector<uint8_t>> images(size, std::vector<uint8_t>(rows * cols));
        for (size_t i = 0; i < size; ++i)
        {
            image_file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
        }

        return { images, labels };
    }

    

private:
    std::string _folder_path;
    std::string _training_images_filepath;
    std::string _training_labels_filepath;
    std::string _test_images_filepath;
    std::string _test_labels_filepath;

    static uint32_t swap_endian(uint32_t val)
    {
        return ((val & 0x000000FF) << 24) |
            ((val & 0x0000FF00) << 8) |
            ((val & 0x00FF0000) >> 8) |
            ((val & 0xFF000000) >> 24);
    }
};
