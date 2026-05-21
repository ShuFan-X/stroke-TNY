import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
#pip install xgboost==2.0.3 --no-deps
#import xgboost
#model = xgboost.Booster()
#model.load_model('XGB.json')
df2 =pd.read_csv('x_test.csv')
x_test = df2[['通脑饮', 'LAA', 'SAO', '年龄', '高血压', '吸烟史', '入院NIHSS', '入院mRS', 'NLR', '淋巴细胞', '尿素', '抗聚史', '降压史', '调脂史']]

model = joblib.load('XGB.pkl')

feature_names = ['通脑饮', 'LAA', 'SAO', '年龄', '高血压', '吸烟史', '入院NIHSS', '入院mRS', 'NLR', '淋巴细胞', '尿素', '抗聚史', '降压史', '调脂史']
    
    
# 设置 Streamlit 应用的标题
st.title("卒中诊断模型")
st.sidebar.header("Selection Panel") # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")
TNY = st.selectbox("通脑饮", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")
LAA = st.selectbox("LAA", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")
SAO = st.selectbox("SAO", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")

AGE = st.number_input("年龄", min_value=0, max_value=120, value=1)

Hypertension = st.selectbox("高血压", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")
Smoke = st.selectbox("吸烟史", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")

NIHSS = st.sidebar.slider("入院NIHSS", min_value=0, max_value=42, value=0, step=1)
mRS = st.sidebar.slider("入院mRS", min_value=0, max_value=42, value=0, step=1)
N = st.number_input("中性粒细胞", min_value=0, max_value=50, value=1)
L = st.number_input("淋巴细胞", min_value=0, max_value=50, value=1)
NLR = N/L
Bun = st.number_input("尿素", min_value=0, max_value=50, value=1)

antiaggregation = st.selectbox("抗聚史", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")
antihypertensive = st.selectbox("降压史", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")
lipid_lowering = st.selectbox("调脂史", options=[0, 1], format_func=lambda x:"否"if x == 1 else "是")






feature_values = [TNY, LAA, SAO, AGE, Hypertension, Smoke, NIHSS, mRS, NLR, L, Bun, antiaggregation, antihypertensive, lipid_lowering]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: 不良结局低风险, 1: 不良结局高风险)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice =(
            f"根据我们的模型，该患者本次卒中预后不良的风险较高。 "
            f"具体预后不良的可能性为 {probability:.1f}%。"
            "建议进一步评估该患者的风险因素，针对性加强预防与治疗干预措施。"
        )

    # 如果预测类别为0（低风险）
    else:
        advice =(
            f"根据我们的模型，该患者本次卒中预后不良的风险较低。"
            f"具体预后不良的可能性为 {probability:.1f}%."
            "但继续保持健康的生活习惯仍是必要的，请定期至医院体检并规律服用药物。"
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP 力图解释")
    explainer_shap = shap.TreeExplainer(model)

    
    explainer = shap.TreeExplainer(model)  # 直接传入模型
    shap_values = explainer.shap_values(x_test)   # 或 explainer(x_test) 返回 Explanation 对象
    shap_values_numpy = shap_values
    #shap_interaction_values = explainer.shap_interaction_values(x_test)
    explanation = shap.Explanation(
    values=shap_values,           # 您的 SHAP 值数组
    base_values=explainer.expected_value,  # 如果是二分类且针对正类，可能需取对应值
    data=x_test.values,            # 特征值（用于着色）
    feature_names=x_test.columns.tolist()
    )

    
    #shap_values = explainer_shap(pd.DataFrame([feature_values], columns=feature_names))
    class_index = 1  # 二分类时正类通常为 1
    expected_value_class = explainer.expected_value[class_index]  # 标量
    shap_values_class = shap_values_numpy[:, :, class_index] 
    if predicted_class == 1:
        #shap.plots.force(explainer_shap.expected_value[1],shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
        shap.force_plot(expected_value_class, 
                        shap_values_class[0, :],           # 一维 SHAP 值
                        pd.DataFrame([feature_values], columns=feature_names),                 # 对应样本的特征值      # 特征名（可选，但建议提供）
                        matplotlib=True,
                        show=True
                       )
    # 期望值（基线值）
    #解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        #shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
        shap.force_plot(expected_value_class, 
                        shap_values_class[0, :],           # 一维 SHAP 值
                        pd.DataFrame([feature_values], columns=feature_names),                 # 对应样本的特征值      # 特征名（可选，但建议提供）
                        matplotlib=True,
                        show=True
                       )    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME 局部解释")
    lime_explainer = LimeTabularExplainer(
        training_data=x_test.values, 
        feature_names=x_test.columns.tolist(),
        class_names=['预后良好', '预后不良'],# Adjust class names to match your classification task
        mode='classification'
    )

    #Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=14
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=True) # Disable feature value table
    st.components.v1.html(lime_html, height=800,scrolling=True)
