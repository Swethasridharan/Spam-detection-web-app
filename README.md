
# 📧 Spam Detection Web App

A simple and interactive spam message detection web application built using **Flask**, **Python**, and **Machine Learning (Naive Bayes Classifier)**.

---

## 🔍 Features

- Input any text message and detect whether it's **Spam** or **Not Spam**
- Clean and preprocessed dataset using TF-IDF
- Real-time classification with trained ML model
- Simple and responsive UI with Flask + HTML

---

## 🧠 Machine Learning Model

- **Algorithm**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF with stop word removal
- **Dataset**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

---

## 🛠️ Technologies Used

- Python
- Flask
- scikit-learn
- Pandas
- HTML/CSS

---

## 🚀 How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/Swethasridharan/Spam-detection-web-app.git
cd Spam-detection-web-app
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
python app.py
```

4. **Visit in browser:**

```
http://127.0.0.1:5000
```

## 📂 Project Structure

```
├── app.py                  # Flask app
├── spam.csv                # Dataset
├── templates/
│   └── index.html          # HTML front-end
└── README.md
```
## ✍️ Author

Swetha Sridharan

## 📄 License

This project is open-source and available under the MIT License.

