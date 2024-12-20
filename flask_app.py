from flask import Flask, request, render_template
from pyngrok import ngrok, conf, exception
import numpy as np
from src.utils import load_model


model = load_model("saves/models/Voting Classifier_20241129_040510.pkl")

app = Flask(__name__)


@app.route("/0")
def index1():
    global model
    if model is None:
        model = load_model("saves/models/Voting Classifier_20241129_040510.pkl")

    features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]

    default_values = {
        "Time": 36855.0,
        "V1": -0.268053912906532,
        "V2": -0.02619367018256,
        "V3": 1.86289362736449,
        "V4": -1.98987575458878,
        "V5": -0.551961334690978,
        "V6": -0.310758677119239,
        "V7": 0.277641458302447,
        "V8": -0.160597607151482,
        "V9": 1.79086900060096,
        "V10": -1.73655779695827,
        "V11": -0.156084709953904,
        "V12": 0.862753826067466,
        "V13": 0.276733057178013,
        "V14": -0.459685649445805,
        "V15": 1.6720175863151,
        "V16": -2.09117437603662,
        "V17": 0.748403680220739,
        "V18": -0.0158527745887944,
        "V19": 1.99438160164543,
        "V20": 0.20607511974538,
        "V21": 0.20158777912739,
        "V22": 1.16262729991667,
        "V23": -0.333368132783458,
        "V24": 0.171281000367694,
        "V25": 0.136556937169513,
        "V26": -0.488835952603244,
        "V27": 0.0634518641078545,
        "V28": -0.0954392660368261,
        "Amount": 25.95,
    }

    # Group features into chunks of four
    feature_groups = [features[i : i + 4] for i in range(0, len(features), 4)]

    return render_template(
        "index.html",
        feature_groups=feature_groups,
        default_values=default_values,
    )


@app.route("/")
def index2():
    features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]

    default_values = {
        "Time": 28658.0,
        "V1": -28.5242675938406,
        "V2": 15.8769229879536,
        "V3": -29.4687320925264,
        "V4": 6.44759140152748,
        "V5": -20.7860000418837,
        "V6": -4.86561341755669,
        "V7": -19.5010840750712,
        "V8": 18.7488719520883,
        "V9": -3.64298981925263,
        "V10": -7.93964241937325,
        "V11": 4.18467368942509,
        "V12": -5.83507521523889,
        "V13": 1.21595964527928,
        "V14": -5.33014378246253,
        "V15": -0.118631138153322,
        "V16": -5.36777527618834,
        "V17": -11.4317569116986,
        "V18": -4.69692444373293,
        "V19": 0.69268841200477,
        "V20": 1.7068890619925,
        "V21": 1.80576978392608,
        "V22": -2.11937616760819,
        "V23": -1.31744962200248,
        "V24": 0.169845630469178,
        "V25": 2.05168733506275,
        "V26": -0.210502000459411,
        "V27": 1.30173396497076,
        "V28": 0.380246181418509,
        "Amount": 99.99,
    }

    # Group features into chunks of four
    feature_groups = [features[i : i + 4] for i in range(0, len(features), 4)]

    return render_template(
        "index.html",
        feature_groups=feature_groups,
        default_values=default_values,
    )


@app.route("/predict", methods=["POST"])
def predict():
    features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]

    input_data = []
    for feature in features:
        value = float(request.form.get(feature, 0.0))
        input_data.append(value)
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction == 1:
        result = "Fraud!"
        result_class = "danger"
    else:
        result = "Legit Transaction"
        result_class = "success"
    return render_template(
    "result.html", prediction=result, result_class=result_class
)


def start_ngrok():
    conf.get_default().auth_token = "NGROK_TOKEN_HERE"
    port = 5000
    try:
        public_url = ngrok.connect(port)
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    except exception.PyngrokNgrokError as e:
        print(f"ngrok connection failed: {e}")
        public_url = None
    return public_url

if __name__ == "__main__":
    public_url = start_ngrok()
    app.run(host="0.0.0.0", port=5000, use_reloader=False, debug=True)
