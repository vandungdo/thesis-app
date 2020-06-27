from wtforms import Form, FloatField, validators, StringField,SelectField,SubmitField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, ValidationError
from wtforms.fields.html5 import DateField
from wtforms_html5 import DateRange 
from wtforms import validators
import pandas as pd
import datetime
import os
import numpy as np

currentPath = os.path.abspath(os.path.dirname(__file__))


df1 = pd.read_csv(os.path.join(currentPath,'data/merged_classes_with_frequency2.csv'))

labels1 = list(df1.label.unique())
labels1.append('all labels')

time1 = list(range(23))
time1.append(-1)
time2 = list(range(1,24))
time2.append(-1)

class NewLabel(Form):

    label = SelectField(label='Label',choices=[(i,i)for i in labels1],default='all labels',validators=[validators.InputRequired()])
    day = DateField('Start date',format='%Y-%m-%d',default = datetime.date(2018,8,1))
    startTime = SelectField(label='Start hour time',choices=[(i,i)for i in time1],default=-1,validators=[validators.InputRequired()],coerce=np.int64)
    endTime = SelectField(label='End hour time',choices=[(i,i)for i in time2],default=-1,validators=[validators.InputRequired()],coerce=np.int64)
    submit = SubmitField('Show',validators=[validators.required()])

    def validate_time(form, field):
        if form.endTime < form.startTime:
            raise ValidationError('end time must not be earlier than start time')

class Algorithms(Form):

    label = SelectField(label='Algorithm', choices = [(i,i) for i in ['SVM','Decision Tree','Naive Bayes','Random Forest']],validators=[validators.InputRequired()])
    submit = SubmitField('Show',validators=[validators.required()])