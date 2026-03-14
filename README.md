# CharityML Donor Prediction App
## A Streamlit web app that predicts whether an individual is likely to be a donor based on their demographics, employment, and financial information. This project demonstrates machine learning deployment and interactive dashboard design using Python.

*🔗 Live Demo*

Check out the app live:
https://supervisedmlproject-nazx6qrbjzmsps5e5naurh.streamlit.app/

Features

Predict donor potential based on input features such as age, education, occupation, marital status, and income.

Interactive Streamlit interface with tabs for demographics, employment, and financial information.

Professional and compact model performance dashboard.

Confidence score displayed for each prediction.

Robust handling of categorical variables for XGBoost models.

Compatible with older XGBoost models using a patch wrapper.

Installation
git clone <your-repo-url>
cd supervisedml_project
pip install -r requirements.txt
Usage
streamlit run app.py

Fill in the form on the left panel (Demographics, Employment, Financial).

Click Predict Donor Potential.

See the prediction result and confidence score.

View model performance metrics on the right panel.

Requirements

Python 3.10

Streamlit

Pandas

NumPy

scikit-learn

XGBoost==1.7.6

joblib

Project Structure
supervisedml_project/
│
├── app.py
├── requirements.txt
├── runtime.txt
└── best_xgb_model.pkl

License

MIT License

## 👥 Contributors

- [Youssef Ebrahim](https://github.com/Youssef-Ebrahim0)
- [Mohamed Nour ](https://github.com/mnour11)
