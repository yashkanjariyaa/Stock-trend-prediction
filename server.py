from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

# Define global variables for the model and training data
model = None

@app.route('/')
def home():
    return render_template('index.html')


model = None
accuracy = None

df = pd.read_csv('data.csv')

# Split dataset into features and target variable
X = df.drop('target_class', axis=1)
y = df['target_class']

# Preprocess categorical variables
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Initialize the RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the classifier
model.fit(X, y)

# Calculate accuracy
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)


@app.route('/predict', methods=['POST'])
def predict():
    global model, accuracy

    if model is None:
        return jsonify({"prediction":"Model not trained yet. Please train the model first by sending a GET request to /train."})

    # for key, value in data.items():
    #     if isinstance(value, pd.Series):
    #         data[key] = value.item()
            
    data = request.json
    
    input_data = pd.DataFrame([data])
    # Make prediction
    prediction = int(model.predict(input_data)[0])

    response = {'prediction': prediction}
    print(f"prediction : {prediction}")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
