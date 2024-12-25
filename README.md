Here's the updated README with author information and a LinkedIn link:

---

# Movie Recommendation AI Model

## Project Overview
This project demonstrates the development of a **Movie Recommendation AI model** using machine learning techniques. The model predicts movie popularity based on various features like genre, budget, runtime, language, and cast details. It utilizes a dataset of over 4,700 movies, exploring key attributes to build a predictive regression model.

---

## Dataset
The dataset contains 21 columns with information such as:
- **Movie_Title**: Title of the movie.  
- **Movie_Genre**: Genre of the movie.  
- **Movie_Budget**: Production budget.  
- **Movie_Revenue**: Revenue generated.  
- **Movie_Vote**: Average user rating.  
- **Movie_Language**: Language of the movie.  

---

## Workflow

### Step 1: Import Libraries
The necessary libraries, such as pandas for data manipulation and scikit-learn for machine learning, are imported:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
```

---

### Step 2: Load Dataset
The dataset is loaded directly from a publicly available URL:
```python
movies = pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Movies%20Recommendation.csv')
print(movies.head())
```

---

### Step 3: Data Preprocessing
- **Target Variable**: `Movie_Popularity`
- **Feature Variables**: Columns such as `Movie_Genre`, `Movie_Language`, `Movie_Budget`, and others are retained. Unnecessary columns like `Movie_ID`, `Movie_Title`, etc., are dropped.
- **One-Hot Encoding**: Categorical variables like `Movie_Genre` and `Movie_Language` are encoded using one-hot encoding for numerical compatibility:
```python
X = pd.get_dummies(X, columns=['Movie_Genre', 'Movie_Language'], drop_first=True)
```

---

### Step 4: Train-Test Split
Split the data into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)
```

---

### Step 5: Handle Missing Values
Handle missing values using the `SimpleImputer` strategy:
```python
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

---

### Step 6: Model Selection and Training
Use a `RandomForestRegressor` for predicting movie popularity:
```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

### Step 7: Prediction
Generate predictions for the test set:
```python
y_pred = model.predict(X_test)
print(y_pred)
```

---

## Key Results
- The model predicts movie popularity as a continuous numerical value.  
- Initial results demonstrate the potential of feature engineering to improve predictions.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/shivanand143/Movie_Recomandation_ai_model.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python movie_recommendation.py
   ```

---

## Future Improvements
- **Incorporate Collaborative Filtering:** Add user-specific interaction data to make recommendations more personalized.  
- **Advanced Models:** Experiment with Gradient Boosting (XGBoost, CatBoost) or Neural Networks for improved accuracy.  
- **Web App Integration:** Build a front-end to make the model accessible to users.

---

## Acknowledgments
- **Dataset Source:** [YBI Foundation's Movie Recommendation Dataset](https://github.com/YBIFoundation/Dataset)  
- **Frameworks:** Powered by Scikit-learn and pandas libraries.

---

## Author
**Shivanand Pujari**  
- 💼 LinkedIn: [Shivanand Pujari](https://www.linkedin.com/in/143shiva)  
- 📧 Email: shivanandpujari666@gmail.com  
- 🐙 GitHub: [@shivanand143](https://github.com/shivanand143)  

Feel free to fork, contribute, or share feedback to make this project even better! 😊

---
