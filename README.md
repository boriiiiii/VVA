# Victory Vision Analytics (VVA)

## Project Overview

Victory Vision Analytics (VVA) is an exciting machine learning-based project aimed at predicting the outcome of Formula 1 races. Using machine learning models, the project analyzes various factors, including driver performance, grid position, and circuit details, to forecast the top contenders for the podium in any given race. The project is powered by **joblib** for model handling and deployed using **Streamlit** for a seamless web interface.

### Features:
- Predict race outcomes based on driver and circuit data.
- Analyze historical race data to improve model predictions.
- Easy-to-use interface powered by Streamlit.

## How to Run the Project

1. **Set up the environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the datamaker**:
    ```bash
    python app.py  # If you have a problem try: python3 app.py

4. **Run the Streamlit app**:
    ```bash
    streamlit run dev.py
