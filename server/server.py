from flask import Flask, request, jsonify
import os
from input_validator import InputSetting
from cespredictor import CESPred
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)

# Load the CES api
model = CESPred()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parser request
        form = dict(request.form)

        # validate the data input and convert the data
        setting = InputSetting(
            form = form,
        )

        # Run CES api
        y_est, y_upper, y_lower = model.predict(setting.form)

        # Return result
        result = {
            'y_est': int(y_est),
            'y_upper': int(y_upper),
            'y_lower': int(y_lower),
        }
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

    except Exception as e:
        response = jsonify({'err': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
