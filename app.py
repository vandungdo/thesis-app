from flask import Flask, render_template, redirect, url_for, request
from werkzeug import secure_filename
from clustering import picture, new_signal_class, picture1, new_df, new_signal_class_time, picture2
from form import NewLabel, Algorithms, PredictionAll
from classification import heatmap
from prediction import after_predict_table_all
from live_prediction import predict_one_file
import datetime
import os
up = os.path.join('static', 'upload')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = up


@app.route('/identification', methods=['GET'])
def identification():

    return render_template('identification.html')


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    newLabelForm = NewLabel(request.form)

    if request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data == -1 and newLabelForm.endTime.data == -1 and newLabelForm.label.data != 'all labels' and newLabelForm.day.data == datetime.date(2018, 8, 1):
        temp1 = new_signal_class(newLabelForm.label.data)
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        pic1 = picture1(newLabelForm.label.data)
    # elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data==-1 and newLabelForm.endTime.data==-1 and newLabelForm.label.data=='all labels' and newLabelForm.day.data==datetime.date(2018,8,22):
    #     temp1 = new_df()
    #     columnNames1 = temp1.columns.values
    #     temp1 = temp1.to_dict('records')
    #     pic1 = []

    elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data != -1 and newLabelForm.endTime.data != -1 and newLabelForm.label.data == 'all labels':
        temp1 = new_signal_class_time(
            newLabelForm.day.data, newLabelForm.startTime.data, newLabelForm.endTime.data)
        labels = list(temp1.label.unique())
        pic1 = picture2(newLabelForm.label.data, newLabelForm.day.data,
                        newLabelForm.startTime.data, newLabelForm.endTime.data)
        # for i in labels:
        #     pic1.append(picture1(i))
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        # pic1 = picture1(newLabelForm.label.data)

    elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data != -1 and newLabelForm.endTime.data != -1 and newLabelForm.label.data != 'all labels':
        temp1 = new_signal_class_time(
            newLabelForm.day.data, newLabelForm.startTime.data, newLabelForm.endTime.data)
        temp1 = temp1[temp1.label == newLabelForm.label.data]
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        pic1 = picture2(newLabelForm.label.data, newLabelForm.day.data,
                        newLabelForm.startTime.data, newLabelForm.endTime.data)

    else:
        columnNames1 = []
        temp1 = {}
        pic1 = []
    pictures1 = []
    for i in pic1:
        pictures1.append(os.path.join(app.config['UPLOAD_FOLDER'], i))
    return render_template('clustering.html', newLabelForm=newLabelForm, colnames1=columnNames1, records1=temp1, pictures1=pictures1)


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    Alg_Form = Algorithms(request.form)
    plots = []
    if request.method == 'POST' and Alg_Form.validate():
        if Alg_Form.label.data == 'SVM':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 85% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' + \
                Alg_Form.label.data + ':'
            plots.append(heatmap(Alg_Form.label.data))
        elif Alg_Form.label.data == 'Decision Tree':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 92% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' + \
                Alg_Form.label.data + ':'
            plots.append(heatmap(Alg_Form.label.data))
        elif Alg_Form.label.data == 'Naive Bayes':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 55% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' + \
                Alg_Form.label.data + ':'
            plots.append(heatmap(Alg_Form.label.data))
        else:
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 93% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' + \
                Alg_Form.label.data + ':'
            plots.append(heatmap(Alg_Form.label.data))
    else:
        accuracy = ''
        lead_sentence = ''
        plots = []
    return render_template('classification.html', Alg_Form=Alg_Form, accuracy=accuracy, lead_sentence=lead_sentence, plots=plots)


@app.route('/prediction_all', methods=['GET', 'POST'])
def prediction_all():
    PredictForm = PredictionAll(request.form)
    cap_all = 'Result of classification all signals in whole dataset'

    if request.method == 'POST' and PredictForm.validate():
        temp_all = after_predict_table_all(PredictForm.day.data, PredictForm.startTime.data,
                                           PredictForm.endTime.data)
        columnNames_all = temp_all.columns.values
        temp_all = temp_all.to_dict('records')

    else:
        columnNames_all = []
        temp_all = {}
    return render_template('prediction_all.html', PredictForm=PredictForm,
                           colnames_all=columnNames_all, records_all=temp_all, cap_all=cap_all)


@app.route('/live_prediction', methods=['GET', 'POST'])
def live_prediction():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        temp = predict_one_file(file_path, request.form.get('band'))
        columnNames = temp.columns.values
        temp = temp.to_dict('records')
        cap = 'Result from live prediction for the file '+filename
    else:
        columnNames = []
        temp = {}
        cap = ''
        f = ''
    return render_template('live_prediction.html', colnames=columnNames, records=temp, cap=cap)


@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')


if __name__ == '__main__':
    app.run(debug=True)
