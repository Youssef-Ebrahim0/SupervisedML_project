# CharityML Donor Prediction App
## A Streamlit web app that predicts whether an individual is likely to be a donor based on their demographics, employment, and financial information. This project demonstrates machine learning deployment and interactive dashboard design using Python.
---

## 🔗 Live Demo

Check out the app live:  
[https://supervisedmlproject-nazx6qrbjzmsps5e5naurh.streamlit.app/](https://supervisedmlproject-nazx6qrbjzmsps5e5naurh.streamlit.app/)

---

## ✨ Features

- Predict donor potential based on features such as **age, education, occupation, marital status, and income**.
- Interactive **Streamlit interface** with tabs for Demographics, Employment, and Financial information.
- **Professional and compact model performance dashboard**.
- Confidence score displayed for each prediction.
- Robust handling of **categorical variables** for XGBoost models.
- Compatible with older **XGBoost models** using a patch wrapper.

---

## 💻 Installation

```bash
git clone <your-repo-url>
cd supervisedml_project
pip install -r requirements.txt
```
## Usage
```bash

streamlit run app.py
```
---

## Project Structure
```bash
supervisedml_project/
│
├── app.py
├── requirements.txt
├── runtime.txt
└── best_xgb_model.pkl
```
---

## 👥 Contributors

- [Youssef Ebrahim](https://github.com/Youssef-Ebrahim0)
- [Mohamed Nour ](https://github.com/mnour11)
