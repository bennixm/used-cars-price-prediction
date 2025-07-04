# Used Cars Price Prediction

This project is a complete machine learning pipeline for predicting the prices of used cars based on various features. It includes data preprocessing, feature engineering, model training (Linear Regression, Decision Tree, Neural Network), hyperparameter tuning, evaluation, advanced visualizations, and interactive user prediction widgets—all in a Jupyter Notebook.

---

## Features

- **Data Cleaning & Exploration:** Handles missing values, explores data, and visualizes distributions.
- **Feature Engineering:** Adds new features like car age.
- **Preprocessing:** Scales numerical features and encodes categorical features using pipelines.
- **Model Training:** Trains and evaluates Linear Regression, Decision Tree, and MLPRegressor (Neural Network).
- **Hyperparameter Tuning:** Uses GridSearchCV for Decision Tree optimization.
- **Evaluation:** Reports MAE, RMSE, R², and visualizes results.
- **Feature Importance:** Visualizes the most important features for tree-based models.
- **Advanced Visualizations:** Correlation heatmap, pairplots, interactive residual plots.
- **User Interaction:** Interactive widgets for making predictions with custom car features.

---

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/used-cars-price-prediction.git
cd used-cars-price-prediction/used_car_prediction
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipywidgets
```

### 3. Prepare the Dataset

Place your `car_dataset.csv` file in the `used_car_prediction` directory.  
The dataset should include columns like brand, model, year, fuel type, transmission, color, price, etc.

### 4. Run the Notebook

Open `used_car_prediction.ipynb` in [Visual Studio Code](https://code.visualstudio.com/) or Jupyter Notebook.

- If using VS Code, install the **Python** and **Jupyter** extensions.
- Select your Python interpreter.
- Run all cells in the notebook.

---

## Usage

- **Explore the data:** The notebook will show data info, statistics, and missing values.
- **Train models:** Models are trained and evaluated automatically.
- **Visualize results:** See actual vs. predicted plots, residuals, feature importances, and more.
- **Try interactive widgets:** Use the interactive cells to filter residuals or predict prices for custom car features.

---

## Example Visualizations

- Correlation heatmap of features
- Pairplot of key numerical features
- Actual vs. predicted price scatterplot
- Residuals distribution and interactive filtering
- Top 10 feature importances for the best model

---

## Customization

- Add or remove features in the preprocessing section as needed.
- Try different models or hyperparameters.
- Extend the notebook with more visualizations or deploy as a web app using Streamlit or Gradio.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [ipywidgets](https://ipywidgets.readthedocs.io/)

---

**Happy Predicting!**