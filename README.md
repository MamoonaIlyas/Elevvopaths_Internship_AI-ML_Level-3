# Elevvopaths_Internship_AI-ML_Level-3
 Task 6:  Music Genre Classification Description

### Project Description

This project, completed as part of the Elevvo AI/ML Internship (Level 3, Task 6), focuses on a multi-class music genre classification task. The primary goal is to build and evaluate machine learning models that can accurately classify audio files into one of ten distinct music genres.

The project explores two different approaches to the problem:
1.  **Tabular Data Approach:** Extracting numerical features from audio files and training a traditional multi-class classifier.
2.  **Image-Based Approach:** Converting audio into visual spectrograms and training a Convolutional Neural Network (CNN) for image classification.

---

### Dataset

The project uses the widely-recognized **GTZAN Dataset** for music genre classification.

* **Source:** [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* **Contents:** 1000 audio files (100 files per genre), each 30 seconds long.
* **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.
* **Note:** The dataset contains a known corrupted file (`jazz.00054.wav`), which is handled during data loading to prevent runtime errors.

---

### Approach & Methodology

#### 1. Tabular Data Classification

This approach treats the audio files as a collection of numerical features.

* **Feature Extraction:** The `librosa` library is used to extract **Mel-Frequency Cepstral Coefficients (MFCCs)**, which are highly effective for representing the timbral and textural qualities of sound. The mean of the MFCCs across each 30-second audio clip is calculated to create a single feature vector per song.
* **Model:** A multi-class classifier from `scikit-learn` is trained on the extracted features. A **Support Vector Machine (SVM)** is a strong baseline model for this task.
* **Evaluation:** The model's performance is evaluated using metrics such as accuracy, precision, and recall via a classification report.

#### 2. Image-Based Classification (CNN)

This approach leverages the power of deep learning for visual pattern recognition.

* **Feature Extraction:** Each 30-second audio file is converted into a **Mel-spectrogram image**, which visually represents the frequency and power of the sound over time.
* **Model:** A **Convolutional Neural Network (CNN)** is built using `Keras` and `TensorFlow`. The CNN is designed to learn hierarchical features and patterns directly from the spectrogram images.
* **Bonus Task (Transfer Learning):** For an advanced approach, a pre-trained model (e.g., VGG16) is used as a feature extractor. The model's final layers are then fine-tuned on the spectrograms, leveraging knowledge learned from large image datasets like ImageNet.

---

### Technologies & Libraries

* **Python:** The core programming language.
* **Librosa:** For loading audio and extracting features (MFCCs, spectrograms).
* **Scikit-learn:** For data preprocessing (scaling, label encoding), data splitting, and training the tabular-based classifier.
* **TensorFlow/Keras:** For building, training, and evaluating the CNN models.
* **Numpy & Pandas:** For efficient data manipulation.
* **Matplotlib:** For visualizing spectrograms.
* **Kaggle API:** For a reliable and reproducible way to download the dataset.

---

### How to Run the Code

The project is designed to be run in a Google Colab environment for convenience and access to GPUs for CNN training.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # or install individually:
    # pip install librosa scikit-learn tensorflow pandas numpy
    ```
3.  **Download the Dataset:**
    * **Authentication:** Go to your Kaggle account, create a new API token (`kaggle.json`), and upload it to your Colab environment when prompted.
    * **Run the data download script:** The main notebook or script will handle the download and unzipping of the GTZAN dataset automatically using the Kaggle API.

4.  **Execute the Notebook/Script:**
    * Follow the code step-by-step to first extract features (MFCCs or spectrograms).
    * Proceed to the respective model training section (Scikit-learn for tabular data or Keras for image data).
    * Evaluate the results and compare the performance of the two approaches.

---

### Conclusion

This task successfully demonstrates the application of both traditional machine learning and deep learning techniques to an audio classification problem. By comparing the results of the tabular and image-based approaches, it provides insight into the strengths and weaknesses of each methodology. The use of the Kaggle API and robust file handling ensures the project is easily reproducible.
