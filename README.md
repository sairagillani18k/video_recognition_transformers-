# 🎥 Video Action Recognition using Transformer Models (ViViT)

##  Project Overview  
This project implements **Video Vision Transformers (ViViT)** for **human action recognition** in video clips.  
Instead of traditional Convolutional Neural Networks (CNNs) or 3D ConvNets, we use **Transformer-based models** to classify actions in video sequences.  

We train a **ViViT model** on the **UCF101 dataset**, a benchmark dataset containing videos of 101 different human activities such as running, playing musical instruments, and various sports.

##  Why Use Transformers for Video Action Recognition?  
Traditional CNN-based methods require complex **3D convolutions** to capture motion over time.  
Transformers, specifically **Vision Transformers (ViTs)**, can model **spatiotemporal dependencies** efficiently through **self-attention mechanisms**.  
By adapting a **pretrained ViT model**, we can significantly reduce training time while achieving state-of-the-art performance.

---

## 📂 Project Structure  

```
📂 video_action_recognition_vivit
│── 📂 models
│   ├── vivit.py  # ViViT model implementation
│── 📂 data
│   ├── dataset.py  # Data preprocessing and loading
│── train.py  # Training script
│── test.py  # Testing script
│── requirements.txt  # Dependencies
│── README.md  # This file
```

---

## 📥 Installation  

### 1️⃣ Install Dependencies  
Ensure Python (>=3.8) is installed, then run:  
```bash
pip install -r requirements.txt
```

### 2️⃣ Clone the Repository  
```bash
git clone https://github.com/YOUR_USERNAME/video_action_recognition_vivit.git
cd video_action_recognition_vivit
```

### 3️⃣ Download UCF101 Dataset  
The dataset is **automatically downloaded** when running the scripts. However, you can also manually download it from:  
[UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)  

---

## 🏋️ Training the Model  

Run the training script:  
```bash
python train.py
```
This will:  
✅ Download UCF101 dataset (if not present)  
✅ Preprocess video frames  
✅ Train the **ViViT model**  

The trained model is saved as `vivit_ucf101.pth`.

---

## 🎯 Testing the Model  

Run the test script:  
```bash
python test.py
```
This will:  
✅ Load the trained model  
✅ Evaluate on the UCF101 test set  
✅ Print the test accuracy  

Example output:  
```
Test Accuracy: 85.42%
```

---

## 🏗 Model Architecture  

### 🔹 What is ViViT?  
**Video Vision Transformer (ViViT)** is an adaptation of **Vision Transformers (ViT)** for **video action recognition**.  
Instead of processing single images like ViT, ViViT takes **multiple frames** and applies **spatiotemporal self-attention** to recognize actions in videos.

### 🔹 How Does It Work?  
1. **Frame Tokenization** - Each video frame is divided into patches.  
2. **Self-Attention Mechanism** - Computes relationships between patches across frames.  
3. **Classification Head** - Outputs a **101-class probability distribution** (for UCF101 dataset).  

---

## 🎬 Dataset: UCF101  

The [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php) is a **benchmark dataset** for human action recognition.  

🔹 **Total Videos**: 13,320  
🔹 **Total Classes**: 101 (e.g., Basketball, Playing Guitar, Skiing)  
🔹 **Dataset Split**:  
   - **Training Set**: ~9,500 videos  
   - **Testing Set**: ~3,800 videos  

---

## Targets for Future Improvements  
 Implement **TimeSformer** for comparison  
 Optimize **training speed** using mixed precision (`torch.cuda.amp`)  
 Experiment with **larger datasets** like Kinetics  

---


