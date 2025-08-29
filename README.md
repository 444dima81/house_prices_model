# 🏠 House Prices Prediction

**Проект выполняется группой студентов буткемпа Эльбрус:**
* Любовь Стрижова
* Кошелев Дмитрий
* Ярослав Пахомов

---

## ℹ️ О проекте
Проект представляет собой веб-приложение на **Streamlit**, которое предсказывает стоимость недвижимости на основе загруженных данных
Используется обученная модель с `scikit-learn` и **CatBoost**, а также пайплайн обработки данных для числовых и категориальных признаков

Пользователь может:  
- 📄 Загрузить CSV с данными о домах  
- 💰 Получить прогноз цены (`salePrice`) для каждой записи или целой таблички
- 📊 Просмотреть статистику прогнозов  

---

## 🛠 Используемые технологии
- Python 3.11.8  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn (Pipeline, StandardScaler)  
- CatBoost  
- Matplotlib

---

## ⚙️ Установка
1. Клонировать репозиторий:  
```bash
git clone https://github.com/yourusername/house-prices-model.git
cd house-prices-model
```
2. Создай виртулаьное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Установи зависимости:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
4. Запусти локально:
```bash
streamlit run main.py
```
`
---

## 📂 Структура проекта
```
house-prices-model/
├── main.py                # Основной скрипт Streamlit
├── pipe_catboost.pkl      # Обученная модель с пайплайном
├── requirements.txt       # Список зависимостей
├── picture/               # Изображения, графики (опционально)
├── notebooks/             # Исходные Jupyter ноутбуки с экспериментами
├── release/               # Финальный ноутбук с готовой моделью
├── data/                  # CSV файлы с данными для обучения/тестирования
└── README.md
