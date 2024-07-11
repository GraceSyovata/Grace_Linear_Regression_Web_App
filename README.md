# Grace_Linear_Regression_Web_App
### Deploying a Linear Regression ML Model as a Web Application on Docker

**Last Updated**: 20 Feb, 2024

### Overview
This project demonstrates how to deploy a Linear Regression machine learning model as a web application using Docker. Docker ensures that the application can run consistently across different environments by using containerization.

### Key Components
1. **Linear Regression**: A supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables.
2. **Docker**: A platform that uses OS-level virtualization to deliver software in containers, which bundle the software with its dependencies and configuration files.

### Steps to Deploy
1. **Train the Linear Regression Model**:
   - **Dataset**: Boston housing dataset.
   - **Features**: Average number of rooms per dwelling (RM), percentage of lower status population (LSTAT), and per capita crime rate by town (CRIM).
   - **Model Training**: Using `scikit-learn`, the model is trained and saved with `pickle`.

2. **Build the Flask Web Application**:
   - **Flask**: Used to create a simple web interface.
   - **Routes**: A form to input data and a route to handle predictions.

3. **Dockerize the Flask App**:
   - **Dockerfile**: Specifies the environment, dependencies, and commands to run the app.
   - **Requirements.txt**: Lists required Python libraries.

4. **Build and Run Docker Image**:
   - **Build Image**: `docker build -t linear-regression-web-app .`
   - **Run Container**: `docker run -p 5000:5000 linear-regression-web-app`

5. **Test the Application**:
   - Access the application via `http://localhost:5000`, input data, and receive predictions.

### Prerequisites
- **Python 3.6+**
- **PIP**
- **Docker**
- **Python Libraries**: Flask, NumPy, Pandas, scikit-learn, Pickle

### Project Structure
```
/your-flask-app
    /templates
        index.html
    app.py
    requirements.txt
    Dockerfile
    ...
```

### Example Code Snippets

**Training the Model**:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
selected_features = data[:, [5, 10, 12]]

X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
```

**Flask Application**:
```python
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='Predicted House Price: ${:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
```

**Dockerfile**:
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

### Conclusion
This project illustrates the process of training a Linear Regression model, developing a Flask web application, and deploying it using Docker. This approach ensures that the ML model is accessible and functional in various environments, making it practical for real-world use.
