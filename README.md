# 💻 LAPTOPIQ – AI-Powered Laptop Price Prediction

**LAPTOPIQ** is a Deep Learning-based Laptop Price Prediction System that estimates laptop prices based on hardware specifications such as brand, processor, RAM, GPU, storage, screen size, weight, and operating system.

Built using Python, TensorFlow, and Streamlit, this project provides an interactive and modern web interface where users can configure laptop specifications and instantly receive an estimated price.

---

## 🚀 Features

- **AI-Powered Prediction:** Uses a trained Artificial Neural Network (ANN) to predict laptop prices.
- **Interactive Web Interface:** Built with Streamlit.
- **Multiple Laptop Brands:** Supports different laptop manufacturers.
- **Hardware Configuration:** Select processor, RAM, GPU, storage, and other specifications.
- **Data Preprocessing:** Uses StandardScaler and One-Hot Encoding.
- **Instant Price Estimation:** Generates predictions based on user inputs.
- **Modern UI:** Premium dark theme with purple, blue, and cyan gradients.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Exploratory data analysis |
| Scikit-learn | Data preprocessing and scaling |
| TensorFlow / Keras | Deep Learning model |
| Streamlit | Interactive web application |
| Pickle | Saving preprocessing objects |

## 🧠 Machine Learning / Deep Learning

The project uses an Artificial Neural Network (ANN) built using TensorFlow and Keras.

**Model Architecture:**
- Input Layer
- Dense Layer (128 neurons, ReLU)
- Dropout (0.2)
- Dense Layer (128 neurons, ReLU)
- Dropout (0.2)
- Dense Layer (64 neurons, ReLU)
- Dropout (0.2)
- Dense Layer (32 neurons, ReLU)
- Dropout (0.2)
- Output Layer for price prediction

**Preprocessing:**
- StandardScaler for numerical features:
  - Inches
  - RAM
  - Weight
- One-Hot Encoding for categorical features.
- Preserved feature ordering using saved model columns.

## 📂 Project Structure

```text
LAPTOPIQ/
│
├── app.py
│
├── Laptop_Price_Prediction.ipynb
│
├── laptop_price_model.keras
│
├── dropdowns.pkl
│
├── model_columns.pkl
│
├── scaler.pkl
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LAPTOPIQ.git
```

### 2. Navigate to the Project Directory

```bash
cd LAPTOPIQ
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## 📊 Input Features

The application accepts the following laptop specifications:

| Feature | Description |
|---|---|
| Company | Laptop manufacturer |
| TypeName | Laptop category |
| Inches | Screen size |
| ScreenResolution | Display resolution |
| Cpu | Processor |
| Ram | RAM capacity |
| Memory | Storage configuration |
| Gpu | Graphics card |
| OpSys | Operating system |
| Weight | Laptop weight |

## 📈 Prediction Output

The application generates an estimated laptop price in Euros (€), based on the model's training target `Price_euros`.

## 🔮 Future Improvements

- Add laptop price comparison functionality.
- Recommend laptops based on user budgets.
- Integrate real-time laptop market prices.
- Add downloadable prediction reports.
- Improve model performance with additional training data.

## 👨‍💻 Author

**Vishal Bhatti**

B.Tech – Artificial Intelligence & Robotics Engineering

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

## ⭐ Support

If you find this project interesting, consider giving this repository a ⭐ on GitHub!

---

*LAPTOPIQ – Discover the true value of your laptop.*
