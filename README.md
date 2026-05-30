# Enhancing Clinical Documentation Accuracy through Generative Pre-Training Powered Natural Language Processing for Medical Speciality Recommendation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20458174.svg)](https://doi.org/10.5281/zenodo.20458174)

This repository contains the official source code and implementation details for the **CMSGPTNLP-MAH** framework, as presented in our paper published in ***The Visual Computer***. 

Our model leverages Generative Pre-training-2 (GPT-2) word embeddings and a Multihead Attention Convolutional Bidirectional Long Short-Term Memory (MA-CBiLSTM) architecture to accurately classify and recommend medical specialties based on clinical text.

## 📊 Dataset
The model is trained and evaluated on the **Healthcare Documentation Database**.
- **Dataset Link:** [Kaggle Healthcare Documentation Database](https://www.kaggle.com/datasets/harshitstark/healthcare-documentation-database)
- **Description:** Contains 2,585 medical transcription samples categorized into 9 distinct medical specialties (e.g., Surgery, Radiology, Neurology, Cardiovascular).

## ⚙️ Dependencies and Requirements
To run the proposed algorithm, your system must meet the following software requirements:
- Python 3.8+
- PyTorch >= 1.12.0
- Transformers (Hugging Face) >= 4.20.0
- scikit-learn
- pandas
- numpy
- nltk

Install all required packages using pip:
```bash
pip install torch transformers scikit-learn pandas numpy nltk
```

## 🧠 Key Algorithms and Pipeline Implementation
The source code follows the exact pipeline detailed in our manuscript:
1. **Text Pre-processing:** Raw medical texts undergo rigorous cleaning, including the removal of special characters/numbers, lowercasing, tokenization, stop word/punctuation removal, and lemmatization/stemming.
2. **Word Embedding (GPT-2):** We utilize the pre-trained GPT-2 transformer model to convert the cleaned clinical tokens into contextualized, high-dimensional numeric vector embeddings.
3. **Classification (MA-CBiLSTM):**
   - **CNN Layers:** Extract local spatial features and semantic clusters from the text sequences.
   - **Bi-LSTM Layers:** Capture bidirectional, long-term contextual dependencies.
   - **Multihead Attention (MA):** Assigns parallel attention weights to different text features, forcing the model to focus on the most relevant context for accurate specialty classification.

## 🚀 Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/CMSGPTNLP-MAH.git
   ```
2. Download the dataset from Kaggle and place it in the `data/` directory as `healthcare_documentation.csv`.
3. Run the training script:
   ```bash
   python main.py
   ```

## 📝 Citation
If you utilize our code or framework in your research, please cite our paper published in **The Visual Computer**:

```bibtex
@article{Alshahrani2024,
  title={Enhancing Clinical Documentation Accuracy through Generative Pre-Training Powered Natural Language Processing for Medical Speciality Recommendation},
  author={Alshahrani, Hussain and Alsudais, Taghreed and Alharbi, Ebtisam and Khalil, Sami F. and Negm, Noha and Alamro, Sulaiman and Almutairi, Nadiyah and Alhashmi, Asma A.},
  journal={The Visual Computer},
  year={2024},
  doi={10.1007/s00371-02X-XXXX-X} 
}
```
