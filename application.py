from flask import Flask, render_template, request, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predictdata", methods=["POST", "GET"])
def predict_data():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        output = predict_pipeline.predict(final_new_data)
        price = round(output[0], 2)

        return render_template("form.html", final_result=price)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

