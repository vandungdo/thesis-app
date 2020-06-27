from flask import Flask, render_template, redirect, url_for,request
from clustering import  picture,new_signal_class,picture1,new_df,new_signal_class_time, picture2
from form import NewLabel, Algorithms
from classification import heatmap
from prediction import before_predict_table,after_predict_table
import datetime
import os
pictureFolder = os.path.join('static','images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = pictureFolder

@app.route('/identification',methods=['GET'])
def identification():
    
    return render_template('identification.html')

@app.route('/clustering',methods=['GET','POST'])
def clustering():
    newLabelForm = NewLabel(request.form)

    if request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data==-1 and newLabelForm.endTime.data==-1 and newLabelForm.label.data!='all labels' and newLabelForm.day.data==datetime.date(2018,8,1):
        print(newLabelForm.day.data)
        print(type(newLabelForm.day.data))
        temp1 = new_signal_class(newLabelForm.label.data)
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        pic1 = picture1(newLabelForm.label.data)
    # elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data==-1 and newLabelForm.endTime.data==-1 and newLabelForm.label.data=='all labels' and newLabelForm.day.data==datetime.date(2018,8,22):
    #     temp1 = new_df()
    #     columnNames1 = temp1.columns.values
    #     temp1 = temp1.to_dict('records')
    #     pic1 = []

    elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data!=-1 and newLabelForm.endTime.data!=-1 and newLabelForm.label.data == 'all labels':
        print(newLabelForm.day.data)
        print('with date and time, all labels')
        temp1 = new_signal_class_time(newLabelForm.day.data,newLabelForm.startTime.data,newLabelForm.endTime.data)
        labels = list(temp1.label.unique())
        pic1=picture2(newLabelForm.label.data,newLabelForm.day.data,newLabelForm.startTime.data,newLabelForm.endTime.data)
        # for i in labels:
        #     pic1.append(picture1(i))
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        # pic1 = picture1(newLabelForm.label.data)

    elif request.method == 'POST' and newLabelForm.validate() and newLabelForm.startTime.data!=-1 and newLabelForm.endTime.data!=-1 and newLabelForm.label.data != 'all labels':
        print(newLabelForm.day.data)
        print('with date and time, and label')
        temp1 = new_signal_class_time(newLabelForm.day.data,newLabelForm.startTime.data,newLabelForm.endTime.data)
        temp1 = temp1[temp1.label == newLabelForm.label.data]
        columnNames1 = temp1.columns.values
        temp1 = temp1.to_dict('records')
        pic1 = picture2(newLabelForm.label.data,newLabelForm.day.data,newLabelForm.startTime.data,newLabelForm.endTime.data)

    else:
        columnNames1 = []
        temp1 = {}
        pic1 = []
    pictures1 = []
    for i in pic1:
        pictures1.append(os.path.join(app.config['UPLOAD_FOLDER'],i))
    return render_template('clustering.html',newLabelForm=newLabelForm,colnames1=columnNames1,records1=temp1,pictures1=pictures1)


@app.route('/classification',methods=['GET','POST'])
def classification():
    Alg_Form = Algorithms(request.form)
    plots = []
    if request.method == 'POST' and Alg_Form.validate():
        if Alg_Form.label.data == 'SVM':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 85% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' +  Alg_Form.label.data +':'
            plots.append(heatmap(Alg_Form.label.data))
        elif Alg_Form.label.data == 'Decision Tree':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 92% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' +  Alg_Form.label.data +':'
            plots.append(heatmap(Alg_Form.label.data))
        elif Alg_Form.label.data == 'Naive Bayes':
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 55% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' +  Alg_Form.label.data +':'
            plots.append(heatmap(Alg_Form.label.data))
        else:
            accuracy = 'The algorithm ' + Alg_Form.label.data + ' has given 93% accuracy rate'
            lead_sentence = 'The plot below shows confusion matrix of the algorithm ' +  Alg_Form.label.data +':'
            plots.append(heatmap(Alg_Form.label.data))
    else:
        accuracy = ''
        lead_sentence = ''
        plots = []
    return render_template('classification.html',Alg_Form = Alg_Form,accuracy = accuracy,lead_sentence=lead_sentence,plots=plots)

@app.route('/prediction',methods=['GET'])
def prediction():
    cap1 = 'Table of signals before prediction'
    temp1 = before_predict_table()
    columnNames1 = temp1.columns.values
    temp1 = temp1.to_dict('records')

    cap2 = 'Table of signals after prediction'
    temp2 = after_predict_table()
    columnNames2 = temp2.columns.values
    temp2 = temp2.to_dict('records')
    return render_template('prediction.html',colnames1=columnNames1,records1=temp1,colnames2=columnNames2,records2=temp2,cap1=cap1,cap2=cap2)

@app.route('/',methods=['GET'])
def welcome():
    return render_template('welcome.html')


if __name__ == '__main__':
    app.run(debug=True)