# 🌸 Flowers Recognition with CNN & Streamlit

This project is a **Deep Learning application** that classifies flower images into 5 categories using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  
It also provides a **Streamlit web app** for interactive predictions.

---

## 📂 Dataset
We use the [Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle.

It contains 5 flower classes:
- Daisy 🌼  
- Dandelion 🌾  
- Rose 🌹  
- Sunflower 🌻  
- Tulip 🌷  

---


## 🚀 Training the Model
You can train the model in Jupyter Notebook or Google Colab.  

Key steps in training:
1. Load dataset from Kaggle.  
2. Split into training and testing sets.  
3. Apply data augmentation (`ImageDataGenerator`).  
4. Define CNN with Conv2D, MaxPooling, BatchNorm, and Dropout.  
5. Train with **Adam optimizer** and **categorical crossentropy loss**.  
6. Save trained model (`flowers.h5` or `flowers.keras`).  

Example command (if training script exists):

  python train_model.ipynb

📊 Results
  Example validation accuracy:

    Final Training Accuracy: ~0.85
    Final Validation Accuracy: ~0.78

🌐 Running the Streamlit App
1. Install dependencies
  Create a virtual environment (recommended) and install requirements:
        pip install tensorflow streamlit pandas matplotlib scikit-learn pillow
2. Run the app
  streamlit run app.py
3. Upload a flower image
    The app resizes the image to 128x128

    Normalizes pixel values (0–1)

    Predicts the flower class using the trained CNN

    Displays the top prediction + probabilities for all 5 classes

🖼 Example Prediction
    Uploaded: sunflower.jpg

    Model output: Sunflower 🌻

    Confidence: 92%

Streamlit also shows a bar chart with probability distribution across all classes.

🔮 Future Work
Use transfer learning with models like VGG16, ResNet50 for better accuracy.

Deploy the Streamlit app to Streamlit Cloud or HuggingFace Spaces.


