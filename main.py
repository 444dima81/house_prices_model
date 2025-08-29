import streamlit as st
import pandas as pd
import numpy as np

# Импорт твоего обученного пайплайна
# Например, pipe_red — это готовый пайплайн после отбора признаков
from joblib import load
pipe_red = load("pipe_catboost.pkl")  # заранее сохрани пайплайн через joblib.dump(pipe_red, "pipe_red.joblib")

st.header("Прогноз цены недвижимости")

# 1) Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV с данными о недвижимости", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Предварительный просмотр данных")
    st.dataframe(df.head())

    # Проверка наличия нужных признаков
    required_features = pipe_red.named_steps['preprocessor'].get_feature_names_out()
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        st.error(f"Отсутствуют необходимые признаки: {missing_features}")
    else:
        # 2) Прогноз
        y_pred_log = pipe_red.predict(df)
        y_pred = np.expm1(y_pred_log)  # возвращаем в исходное пространство

        df_result = df.copy()
        df_result["PredictedPrice"] = y_pred

        st.subheader("Результаты прогноза")
        st.dataframe(df_result.head())

        st.subheader("Статистика прогнозов")
        st.write(df_result["PredictedPrice"].describe())