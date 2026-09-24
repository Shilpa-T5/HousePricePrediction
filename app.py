from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained ML model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])

    # Make prediction
    prediction = model.predict([[area, bedrooms, bathrooms]])

    price = prediction[0]

    return render_template(
        "index.html",
        prediction=f"₹ {price:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)