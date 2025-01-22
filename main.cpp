#include <iostream>

#include <SFML/Graphics.hpp>

#include "trainer.hpp"
#include "drawer.hpp"

int main()
{
    std::string folder_path = "SPECIFY YOUR FOLDER PATH HERE";
    std::string training_images_filepath = "train-images.idx3-ubyte";
    std::string training_labels_filepath = "train-labels.idx1-ubyte";
    std::string test_images_filepath = "t10k-images.idx3-ubyte";
    std::string test_labels_filepath = "t10k-labels.idx1-ubyte";
    
    Trainer nr{ folder_path, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath };
    auto& nn = nr.Run();

    Drawer dr;
    dr.Run(nn);
}