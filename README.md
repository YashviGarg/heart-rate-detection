# Heart Rate Monitoring System using Human Speech

**B.Tech Final Year Project (Aug 2021 - Dec 2021)**

## Overview

This patent-pending deep learning project uses speech signals to classify heart rates into normal and abnormal categories, accounting for both physical and emotional factors. The system leverages advanced audio processing techniques and neural networks to provide a non-invasive approach to heart rate monitoring and classification.

## Key Achievements

- Developed a novel approach to heart rate monitoring using only speech signals
- Achieved **79% accuracy and 0.89 precision** in heart rate classification
- Realized an **80% increase in information retrieval efficiency** through audio preprocessing techniques
- Created a system that accounts for physical and emotional factors affecting heart rate

## Features

- **Advanced Audio Preprocessing**: Loads and processes speech recordings using specialized techniques
- **Feature Extraction**: Extracts MFCCs and mel-spectrograms from audio data with Librosa
- **Data Augmentation**: Implements four different audio augmentation techniques:
  - Random noise addition
  - Time shifting
  - Pitch modification
  - Speed adjustment
- **Multi-label Classification**: Classifies heart rates while accounting for:
  - Gender (Male/Female)
  - Age group (20-29, 30-39, 40-49, 50-59, 60-69)
  - Emotional state (after_workout, happy, neutral, relaxed, stressed, tired)
  - BPM (heart rate in beats per minute)
- **Comprehensive Model Evaluation**: Performance metrics including accuracy, precision, recall, and F1-score

## Technical Implementation

The project uses the following libraries and technologies:

- **Librosa**: For audio feature extraction and manipulation
- **TensorFlow/Keras**: For building and training the neural network
- **Pandas & NumPy**: For data manipulation and numerical operations
- **Scikit-learn**: For data preprocessing and model evaluation
- **Matplotlib**: For visualization of audio signals and spectrograms

The neural network architecture consists of:
- Input layer accepting 26 features
- Three hidden layers with ReLU activation and dropout for regularization
- Output layer with softmax activation for multi-class classification

## Dataset

The model is trained on a diverse dataset of speech signals:
- Professionally annotated by a medical expert
- Includes subjects of different genders, age groups, and emotional states
- Each recording is labeled with these attributes along with heart rate measurements
- Available via [Google Drive](https://drive.google.com/drive/folders/19tc65jlCDst04DHeCmG3SAtVSIXoeHVW?usp=sharing)

## Usage

1. **Prepare the Dataset**: Download the speech dataset from the Google Drive link and place it in the `project/testCases/` directory
2. **Update Metadata**: Ensure the `project/final_dataset.csv` contains proper information about your recordings
3. **Run the Model**: Execute the script to extract features, train the model, and evaluate performance:
   ```bash
   python finalyearclassification.py
   ```
4. **Access Results**: The trained model will be saved at `saved_models/audio_classification.hdf5`

## Requirements

- Python 3.9.16
- TensorFlow 2.0+
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Future Work

Potential improvements and extensions include:
- Implementing more sophisticated feature extraction techniques
- Exploring different neural network architectures (e.g., CNNs, RNNs)
- Expanding the dataset with more diverse speech recordings
- Deploying the model as a web application or mobile app for practical use

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute this code for academic, personal, or commercial purposes. Attribution is appreciated but not required.

## Contact

For any questions or feedback, feel free to reach out:
- **Email**: [yashvigarg8080@gmail.com](mailto:yashvigarg8080@gmail.com)
- **LinkedIn**: [Yashvi Garg](https://www.linkedin.com/in/yashvigarg)
- **GitHub**: [Yashvi Garg](https://github.com/YashviGarg)