import os
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify

import numpy as np
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from database import *
import random

from googletrans import Translator 
import os
import tempfile
from datetime import datetime
from pre import *
 
# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
@app.route('/')
def m():
    return render_template('index.html')
@app.route('/al')
def m3():
    return render_template('alogin.html')

@app.route('/ar')
def m4():
    return render_template('areg.html')

@app.route('/uh')
def m5():
    return render_template('ahome.html')
@app.route('/uf')
def m6():
    return render_template('vupload.html')
@app.route('/uf1')
def m61():
    return render_template('iupload.html')
@app.route('/uf2')
def m62():
    return render_template('tupload.html')
@app.route('/l')
def logout():
    return render_template('index.html')  # Replace with logic for logout if needed
@app.route("/Admin_register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = acc_reg(username,email,password)
        if status == 1:
            return render_template("/alogin.html")
        else:
            return render_template("/areg.html",m1="failed")        
    

@app.route("/admin_login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = acc_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1: 
            session['username'] = request.form['username']                                     
            return render_template("/ahome.html", m1="sucess")
        else:
            return render_template("/alogin.html", m1="Login Failed")


 
@app.route('/predict_text', methods=['POST'])
def handle_predict_text():
    input_text = request.form['text']
    language = request.form['language']

    # Translate text to English
    translator = Translator()
    translated_text = translator.translate(input_text, src=language, dest='en').text
    print(translated_text)
    # Predict based on translated text
    prediction = predict_text(translated_text)

    return render_template('tupload.html', text=prediction)
 

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)