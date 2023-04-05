import dataclasses as dtcls
import numpy as np
import pandas as pd
import chardet
import json
import csv
import os 
import shutil
import codecs

from scipy.stats import uniform, randint
from base64 import encode
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from io import StringIO
from flask import Flask, redirect, render_template, request, send_file, send_from_directory
from lib import transcat

from lib.storage.csv import CSVTrainStorage, CSVUpdateStorage
from lib import transcat


app = Flask(__name__)
app.config["DEBUG"] = True


# Upload folder
load_dotenv(Path.joinpath(Path.cwd(), ".env"))


if os.environ.get("UPLOAD_FOLDER"):
    UPLOAD_FOLDER =  Path.joinpath(Path.cwd(), os.environ.get("UPLOAD_FOLDER")) 
else:
    UPLOAD_FOLDER =  '/code/files'

if os.environ.get("DOWNLOAD_FOLDER"):
    DOWNLOAD_FOLDER =  Path.joinpath(Path.cwd(), os.environ.get("DOWNLOAD_FOLDER")) 
else:
    DOWNLOAD_FOLDER =  '/code/download'


app.logger.debug(f"{UPLOAD_FOLDER=}")
app.logger.debug(f"{DOWNLOAD_FOLDER=}")

app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] =  DOWNLOAD_FOLDER


def load_csv(csvFilePath):
    jsonArray = []

    f = open(csvFilePath,"rb").read()

    asdf = chardet.detect(f)

    #read csv file
    with open(csvFilePath, encoding=asdf['encoding']) as csvf: 
        
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
    return jsonArray
    
    
@app.route('/', methods=['GET'])
def index():
    onlyfiles = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]

    if "tmp.csv" in onlyfiles:
        onlyfiles.remove("tmp.csv")

    return render_template('index.html', data=onlyfiles)

@app.route('/remove_model', methods=['POST'])
def remove_model():

    feture_vec = list( request.form.to_dict().values() )
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    os.remove(file_path)
    
    return redirect("/")

@app.route('/load_model', methods=['POST'])
def load_model():

    feture_vec = list( request.form.to_dict().values() )
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    
    data = load_csv( file_path)

    return render_template('model.html', data={
        "data": data,
        "model_name": feture_vec[0]
    })

# Get the uploaded files
@app.route("/upload", methods=['POST'])
def uploadFiles():

    uploaded_file = request.files['file']
    sep = request.form['seperator']
    linestart = request.form['linestart']

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
    
    f = open(file_path,"rb").read()

    codec = chardet.detect(f)

    BLOCKSIZE = 1048576 
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp.csv")
    with codecs.open(file_path, "r", codec['encoding']) as sourceFile:
        with codecs.open(tmp_name , "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
    os.remove(file_path)

    with codecs.open(tmp_name, 'r', "utf-8") as fin:
        data = fin.read().splitlines(True)
        
    with codecs.open(tmp_name, 'w', "utf-8") as fout:
        fout.writelines( [ e for e in data[int(linestart):] ])

    asdd = pd.read_csv(tmp_name, sep=sep)
    asdd.drop(asdd.filter(regex="Unname"),axis=1, inplace=True)

    if "Kategorie" in asdd:
        df_y = pd.DataFrame(asdd, columns=['Kategorie']).values.tolist()
        df_y_flat = list(dict.fromkeys( [subitem for item in df_y for subitem in item] ))
        y_existing_str = asdd["Kategorie"]
        asdd.drop(["Kategorie"], axis=1, inplace=True)
    else:
        df_y_flat = []
        y_existing_str = []

    asdd.to_csv(tmp_name, index=False)

    return render_template('label.html', data={
        "lable_predict": y_existing_str,
        "lable_predict_single": [ e for e in df_y_flat],
        "data": load_csv(tmp_name)
    })

@app.route('/save_model', methods=['POST'])
def save_model_create():
    feture_vec = list( request.form.to_dict().values() )

    model_name = feture_vec[0] + ".csv"
    feture_vec = feture_vec[1:]

    final_name = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp.csv")

    df_pred = pd.read_csv(tmp_name, delimiter=',')

    df_pred.insert(len(df_pred.columns), "Kategorie", feture_vec)

    df_pred.to_csv(final_name, sep=",", index=False)

    onlyfiles = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]

    onlyfiles.remove("tmp.csv")

    return render_template('index.html', data=onlyfiles)


@app.route('/save', methods=['POST'])
def save_model_predict():
    feture_vec = list( request.form.to_dict().values() )

    train_model = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    predict_model_down_filename = feture_vec[1]
    predict_model = os.path.join(app.config['UPLOAD_FOLDER'], predict_model_down_filename)
    predict_model_down = os.path.join(app.config['DOWNLOAD_FOLDER'], predict_model_down_filename)

    shutil.move(predict_model, predict_model_down)

    storage = CSVUpdateStorage(
        train_model,
        predict_model_down,
    )

    feture_vec = feture_vec[2:]

    print(feture_vec)

    transcat.save_model_predict(storage, feture_vec)

    return send_file(predict_model_down)


@app.route('/train', methods=['POST'])
def train_model_predict():
    feture_vec = list( request.form.to_dict().values() )

    uploaded_file = request.files['form_file']

    form_sep = feture_vec[0]
    form_linestart = feture_vec[2]
    form_model_name = feture_vec[3]

    feture_vec = feture_vec[4:]

    ### --- Save the file ---

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_"+uploaded_file.filename)
        uploaded_file.save(file_path)
    
    storage = CSVTrainStorage(
        file_path,
        os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename),
        os.path.join(app.config['UPLOAD_FOLDER'], form_model_name),
        int(form_linestart),
        form_sep,
    )

    res = transcat.train_model_predict(storage, feture_vec)

    os.remove(file_path)

    return render_template('train.html', data={
        "fet_vec": feture_vec,
        "data": load_csv(storage.tmp_name),
        "train_model_name": form_model_name,
        "predict_model_name": uploaded_file.filename,
        **dtcls.asdict(res),
    })

@app.route('/asdf/', methods=['GET'])
def fetch():
    data = load_csv('/code/train.csv')

    return render_template('train.html', data=json)
