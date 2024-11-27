
# Audio-Dealization using Deep Learning

## Overview
This project focuses on **Audio Dealization**, which involves identifying the speaker at specific timestamps in an audio file and transcribing their speech. The objective is to recognize and label the speakers along with time stamps, using deep learning models for speaker detection.

Used presidential election data. The data is in the form of .mp3 audio file and .csv file. We convert the .mp3 audio file into .wav format and trimed it to a shorter length to get better understanding of the data. Created the time series for the following data to create a PCA to find the speakers for the specific time.Then we trained and tested the model with a CNN model to predict and to see how accurate the model performed. For that we calculate the confusion matrix as well as classification report.

### Key Features:
- **Speaker Identification**: Identifies different speakers in an audio file.
- **Timestamps**: Provides time-based labels for each speaker's speech segment.
- **Deep Learning Model**: Uses Convolutional Neural Networks (CNN) for speaker prediction and transcription.

## Setup and Requirements

### Prerequisites:
1. Python 3.x
2. Required libraries:
   - `numpy`
   - `pandas`
   - `librosa`
   - `matplotlib`
   - `scikit-learn`
   - `tensorflow` (for CNN model)

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset:
This project uses `.mp3` audio files and `.csv` files containing the transcription data. The audio is converted to `.wav` format and trimmed for better analysis. 

### Instructions:

1. **Convert MP3 to WAV**:
   - Use `librosa` or other tools to convert audio files to `.wav` format.
   
2. **Data Preprocessing**:
   - The data is segmented and timestamps are created for PCA analysis.
   - Extract features such as **MFCCs** (Mel-frequency cepstral coefficients) for use in the CNN model.

3. **Model Training**:
   - The CNN model is trained on the processed audio features to predict speakers and time stamps.
   - The model achieves an accuracy of approximately **61%**.

4. **Results**:
   - After training, the model evaluates the predictions and calculates the **confusion matrix** and **classification report**.

### Example Usage:
```bash
python train_model.py
```

### Evaluation:
After the model is trained, use the following to test it:
```bash
python evaluate_model.py
```

This will print out the classification report and confusion matrix, helping assess the model's performance.

---

## Detailed Documentation

### 1. Data Preprocessing
Data is converted from `.mp3` to `.wav`, followed by trimming the length for better model training. We perform **time series analysis** to create a Principal Component Analysis (PCA) model, which helps identify the speakers at specific timestamps.

### 2. Feature Extraction
For each audio segment, features such as **MFCCs** are extracted. These features are then used for training a Convolutional Neural Network (CNN) model, which learns to associate these features with speaker identity and the corresponding time frames.

### 3. Model Architecture
The model uses a simple CNN to classify audio segments based on extracted features. The model consists of several convolutional layers followed by dense layers, ultimately outputting the predicted speaker for each segment.

### 4. Performance Evaluation
The performance of the trained model is evaluated using metrics such as **accuracy**, **confusion matrix**, and **classification report**. The model achieves an accuracy of **61%** on the test data.

### 5. Future Improvements
- Explore more advanced deep learning models for higher accuracy.
- Integrate **Natural Language Processing (NLP)** for better transcription.
- Use more diverse datasets for better speaker recognition.

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- The dataset used in this project is based on speech data for speaker identification.
- Libraries like **librosa**, **tensorflow**, and **scikit-learn** were used for audio processing, model training, and evaluation.
