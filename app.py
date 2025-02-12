from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app with correct directories
app = Flask(__name__, static_folder="static", template_folder="templates")

# Fix serving of static files (CSS, JS, Images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Load dataset
dataset_path = os.path.join(os.getcwd(), "dataset", "mail_data.csv")
df = pd.read_csv(dataset_path)
df = df.where((pd.notnull(df)), '')

# Convert labels: spam = 0, ham = 1
df.loc[df['Category'] == 'spam', 'Category'] = 0
df.loc[df['Category'] == 'ham', 'Category'] = 1

# Split data
X = df['Message']
Y = df['Category'].astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Fix the root route to render the correct HTML page
@app.route('/')
def home():
    return render_template('index.html', email_text="")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['email_text'] 
    input_features = vectorizer.transform([user_input])
    prediction = model.predict(input_features)[0]

    result = "Not Spam" if prediction == 1 else "Spam"
    return render_template('index.html', email_text=user_input, prediction=result)

# Run Flask app on port 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)
