# Kabyle Speech-to-Text Model (PyTorch)

[![Kabyle ASR](https://img.shields.io/badge/ASR-Kabyle-blue)](https://github.com/alexsaad80/kabyle-speech2text)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Operational-green)](https://github.com/alexsaad80/kabyle-speech2text)

This project provides a speech recognition model (STT - Speech To Text) for the Kabyle language.
The pre-trained model `best_kabyle_asr_optim.pt` was trained on more than 700,000 audio sentences with their textual transcriptions, from Common-voice and Tatoeba.

This model was obtained after xx Epochs. The script saves the best model version according to the 'Val Loss' value and after ten times without improvement of this value, the script will stop and keep the last best version.


## 📋 Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Testing the Pre-trained Model](#testing-the-pre-trained-model)
  - [Custom Training](#custom-training)
- [Recommended CUDA Configuration](#recommended-cuda-configuration)
- [Best Practices and Tips](#best-practices-and-tips)

## 📚 Overview

The Kabyle speech recognition model has been designed to transform audio recordings in Kabyle into written text. The optimized model is available under the name `best_kabyle_asr_optim.pt`.

> **Note**: Using a GPU is strongly recommended for optimal performance.

## 🖥️ Requirements

- Python 3.8+ 
- PyTorch
- CUDA-compatible GPU (strongly recommended)
- Sufficient disk space for training data

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/alexsaad80/kabyle-speech2text.git
cd kabyle-speech2text
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Testing the Pre-trained Model

1. Run the test script:
```bash
python kab_audio_model_test.py
```

2. A Gradio interface will be generated - follow the link displayed in the console
3. Upload your MP3 audio file via the interface to get the transcription

### Custom Training

To train the model with your own audio files:

1. Place your MP3 files in the `audios/` folder

2. Update the `transcription.csv` file with your information, following this format:

   You can check the current "audios" folder and "transcription.csv" file for a sample
```
audio_file_name;transcription
common_voice_kab_21869653.mp3;Ssnent ad wtent apyanu
common_voice_kab_21869655.mp3;Iḥeqqa tekkateḍ imẓad
common_voice_kab_21869656.mp3;Lliɣ ssedhuyeɣ igerdan
```

3. Launch the training:
```bash
python train_kab.py
```

## ⚙️ Recommended CUDA Configuration

For optimal performance, ensure your CUDA installation is compatible:
- NVIDIA driver ≥ 530.xx
- CUDA Toolkit 12.1 (for cu121)

To check your CUDA version:
```bash
nvcc --version
```

## 💡 Best Practices and Tips

- **Data Volume**: A minimum of 5,000 audio files with their transcriptions is necessary to obtain a coherent model.
- **Data Format**: 
  - Use a semicolon (`;`) as a separator between the audio filename and its transcription
  - Replace any semicolons with spaces in your sentences before integrating them into the transcription file
- **Performance Optimization**: Adjust the script parameters according to your GPU capabilities:
  - `BATCH_SIZE`
  - `num_worker`
  - `prefetch_factor`

---

*Project developed for the preservation and automatic processing of the Kabyle language.*

[GitHub Repository](https://github.com/alexsaad80/kabyle-speech2text)
