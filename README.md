# Heart Rate Monitoring System using Human Speech

## Overview

This repository contains a deep learning project designed to classify heart rates as normal or abnormal from human speech signals. The model utilizes advanced audio preprocessing techniques and a neural network to analyze speech signals and predict heart rate categories with high accuracy.

## Features

- **Deep Learning Model**: Built using TensorFlow and PyTorch, capable of classifying heart rates based on speech signals.
- **Audio Preprocessing**: Leverages the Librosa library for comprehensive audio feature extraction.
- **Data Augmentation**: Includes data augmentation techniques to enhance model performance.
- **High Accuracy**: Achieves an accuracy of 79% and a precision of 0.89 on the test dataset.

## Project Structure

```bash
Project/
├── finalyearclassification.py  # Main script for model implementation
├── data/                       # (Optional) Directory for storing the dataset
├── models/                     # (Optional) Directory for saving trained models
├── logs/                       # (Optional) Directory for storing logs
├── .gitignore                  # Configuration for ignoring unnecessary files
└── README.md                   # Project documentation (this file)
```

## Getting Started

### Prerequisites

Ensure you have Python 3.9.16 installed. You'll also need the following Python packages:

- TensorFlow
- PyTorch
- Librosa
- NumPy
- Pandas
- Scikit-learn

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Usage

Ensure you have Python 3.9.16 installed. You'll also need the following Python packages:

1. **Prepare the Dataset**: Place your annotated speech dataset in the `data/` directory.

2. **Train the Model**: Run the following command to start the training process:
```bash
python finalyearclassification.py
```
3. **Evaluate the Model**: The script will automatically evaluate the model's performance on the test dataset and display the accuracy and precision metrics.

## Results

The model has been tested and achieves the following results:

- **Accuracy**: 79%
- **Precision**: 0.89

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request for review.

## License

This project is open for viewing purposes only. No license is granted for use, modification, or distribution of this code. Please contact me directly if you have any questions.

## Contact

For any questions or feedback, feel free to reach out:

- **Email**: [yashvigarg8080@gmail.com](mailto:yashvigarg8080@gmail.com)
- **LinkedIn**: [Yashvi Garg](https://www.linkedin.com/in/yashvigarg)
- **GitHub**: [Yashvi Garg](https://github.com/YashviGarg)
