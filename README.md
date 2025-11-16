# 🌾 Wheat Crop Disease Classification (Multi-Label Classification)

This project presents a deep learning-based approach to detect multiple wheat leaf diseases from images. It uses a *multi-label* classification pipeline that identifies one or more diseases or a healthy label in a single image. Additionally, it provides insights model explainability with grad-CAM.

View project demo here: [Demo Link](https://huggingface.co/spaces/OmkarDhekane/wheatCropClassifier)<br>
All trained models used during the experiments can be found here: [Model Link](https://www.kaggle.com/models/omkardhekane/wcdc_trial)<br>
The Data is hosted on my Kaggle account here: [Data Link](https://www.kaggle.com/datasets/omkardhekane/my-seminar-dataset)<br>
_Note that, since data is private, its stored on kaggle private repo. To access it, you may need to send a request. please send me a email to request data preview._

## 📌 Overview

- **Type**: Multi-label Image Classification  
- **Model**: DenseNet121, VGG19, EfficientNetB1, MobileNetV2, InceptionV3
- **Dataset Size**: 2,414 (original) + 675(augmented) ~3K images 
- **Classes**: 6 Diseases + 1 Healthy  
- **Goal**: Robust and accurate classifier of crop diseases to assist early intervention.
- **Best Model**: DenseNet121
- **Best Strategy**: Data Expansion + Augmentation + Finetuning
- **Results**: See *results.csv* file
---


## 🛠️ Setup and Usage

### 🔧 1. Clone and Install
```bash
git clone https://github.com/OmkarDhekane/wcdc.git && cd wcdc
```

Create virtual enviornment:
```bash
conda create -n wheat-app python=3.11 -y
conda activate wheat-app
```
Install required dependencies:
```bash
pip install -r requirements.txt
```

If you want to run the application locally, run the command
```bash
streamlit run app.py
```

### 🔍 2. Run Notebook 

optionally, you can also run the `.ipynb` file to run the experiments.

---

📬 Contact
Author: Omkar Dhekane <br>
📧 Email: dhekane2@illinois.edu <br>


💡 Early disease detection can help save yields, reduce pesticide use, and promote sustainable agriculture.
