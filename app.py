import numpy as np
import joblib  # To load the trained model
import streamlit as st

# Load the trained scaler and classifier
scaler = joblib.load('scaler.pkl')
classifier = joblib.load('classifier.pkl')

# Streamlit UI
def main():
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

    st.title("🔍 Diabetes Prediction App")
    st.markdown("### Enter the following details to check for diabetes risk:")

    # User input fields with side-by-side layout
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, step=1, help="Number of times pregnant")
        Glucose = st.number_input("🩸 Glucose Level", min_value=0, max_value=200, step=1, help="Blood glucose concentration")
        BloodPressure = st.number_input("💓 Blood Pressure", min_value=0, max_value=150, step=1, help="Diastolic blood pressure")
        SkinThickness = st.number_input("📏 Skin Thickness", min_value=0, max_value=100, step=1, help="Triceps skinfold thickness")

    with col2:
        Insulin = st.number_input("💉 Insulin", min_value=0, max_value=900, step=1, help="Serum insulin in mu U/ml")
        BMI = st.number_input("⚖️ BMI", min_value=0.0, max_value=60.0, step=0.1, help="Body mass index")
        DiabetesPedigreeFunction = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, help="Diabetes hereditary risk")
        Age = st.number_input("🎂 Age", min_value=0, max_value=120, step=1, help="Age in years")

    # Centered button
    st.markdown("---")
    if st.button("🚀 Predict", use_container_width=True):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        std_data = scaler.transform(input_data)
        prediction = classifier.predict(std_data)

        if prediction[0] == 0:
            st.success("✅ The person is **not diabetic**. Keep maintaining a healthy lifestyle! 🏃‍♂️")
        else:
            st.error("⚠️ The person is **diabetic**. Please consult a doctor for further advice. 🏥")

    # Footer
    st.markdown("---")
    st.markdown("📌 **Note:** This prediction is based on machine learning and should not replace professional medical advice.")

if __name__ == "__main__":
    main()


