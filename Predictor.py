import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

df2 = pd.read_csv('x_test.csv')

x_test = df2[['NLR', 'admissionNIHSS', 'ePWV', 'Glu', 'Drinking']]

model = joblib.load('XGB.pkl')

feature_names = [
    "admissionNHISS",
    "Drinking",
    "ePWV",
    "NLR",
    "Glu"
]

# 设置 Streamlit 应用的标题
st.title("Prospective study with a 90-day follow-up")
st.sidebar.header("Selection Panel")  # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")
admissionNHISS = st.number_input("admissionNHISS", min_value=0, max_value=42, value=0)
# admissionNHISS = st.sidebar.slider("admissionNHISS", min_value=0, max_value=42, value=0, step=1)
Drinking = st.selectbox("Drinking", options=[0, 1], format_func=lambda x: "Drinking" if x == 1 else "no Drinking")
age = st.number_input("age", min_value=0, max_value=120, value=0)
# age = st.sidebar.slider("age", min_value=0, max_value=120, value=0, step=1)
SBP = st.number_input("SBP", min_value=0, max_value=300, value=0)
# SBP = st.sidebar.slider("SBP", min_value=0, max_value=300, value=0, step=1)
DBP = st.number_input("DBP", min_value=0, max_value=300, value=0)
# DBP = st.sidebar.slider("DBP", min_value=0, max_value=300, value=0, step=1)
N = st.number_input("N", min_value=0, max_value=50, value=0)
L = st.number_input("L", min_value=0, max_value=50, value=0)
Glu = st.number_input("Glu", min_value=0, max_value=50, value=0)

MBP = DBP + 0.4 * (SBP - DBP)
ePWV = 9.587 - 0.402 * age + 4.56 * 0.001 * age * age - 2.621 * 0.00001 * age * age * MBP + 3.176 * 0.001 * age * MBP - 1.832 * 0.01 * MBP
NLR = N / L

feature_values = [admissionNHISS, Drinking, ePWV, NLR, Glu]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class} (1: Bad Prognosis, 0: Good Prognosis)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Bad Prognosis. "
            f"The model predicts that your probability of having heart disease is (probability:.1f)%."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of Bad Prognosis. "
            f"The model predicts that your probability of not having heart disease is (probability:.1f)%."
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value[e], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=x_test.values,
        feature_names=x_test.columns.tolist(),
        class_names=['Good Prognosis', 'Bad Prognosis'],  # Adjust class names to match your classification task
        mode='classification'
    )

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)