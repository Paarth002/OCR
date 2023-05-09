import os
from flask import Flask, render_template, redirect, url_for, request
import cv2
import imutils
import numpy as np

UPLOAD_FOLDER = './static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

file_name = ''

def scan_text(cnts, method="l2r"):
    reverse = False
    i = 0
    if method == "r2l" or method == "b2t":
        reverse = True
    if method == "t2b" or method == "b2t":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

from keras.models import load_model
model = load_model('./models/ocr-model.h5')
model2 = load_model('./models/ocr-model-2.h5')

classes1 = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
           9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 
           17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 
           25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 
           33: 'X', 34: 'Y', 35: 'Z'}

classes2 = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
            9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 
            17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 
            25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 
            33: 'Y', 34: 'Z'}

def get_letters(img, model_num=1):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = scan_text(cnts, method="l2r")[0]
    # loop over the contours
    letters = []
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)
        if model_num==1:
          [y_pred] = model.predict(thresh)
          y_pred = y_pred.argmax()
  #         print(y_pred)
          pred = classes1[y_pred]
          print(pred, y_pred)
          # letters.append(pred)
        else:
          [y_pred2] = model2.predict(thresh)
          y_pred2 = y_pred2.argmax()
  #         print(y_pred)
          pred2 = classes2[y_pred2]
          # print(pred2, y_pred2)
          letters.append(pred2)

    letters_f = ""
    for i in letters:
        letters_f += (i + " ")
    return image, letters_f

global final_result
final_result = ''

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        file1_type = file1.content_type
        if file1_type != 'image/jpeg' and file1_type != 'image/png':
            return 'Please upload image!'
        
        global file_name
        file_name = "orig."+file1.content_type.split('/')[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file1.save(file_path)
        image, letters = get_letters(file_path, 2)
        cv2.imwrite("static/result.jpg", image)
        global final_result
        final_result = ''
        final_result = letters
        return redirect(url_for('result'))  
    
    return render_template('classify.html')


def get_words_ocr(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = scan_text(cnts, method="l2r")[0]
    # loop over the contours
    all_words = []
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        all_words.append(thresh)

    return all_words, image

def get_positions_ocr(vpp):
  ps = []
  cur = [-1, -1]
  
  for i in range(len(vpp)):
    if vpp[i] == 0:
      if cur != [-1, -1]:
        ps.append(cur)
        cur = [-1, -1]
    else:
      if cur[0] == -1:
        cur[0] = i
      else:
        cur[1] = i
      
  if vpp[-1] != 0:
    ps.append(cur)
  
  return ps

def get_letters_ocr(words, model_num = 2):
  all_letters = []
  for w in words:
    vpp = np.sum(w, axis = 0)
    print(vpp)
    char_pos_list = get_positions_ocr(vpp)

    letters = []
    for char_pos in char_pos_list:
      char = w[:, char_pos[0]:char_pos[1]+1]
      thresh = cv2.resize(char, (32, 32), interpolation = cv2.INTER_CUBIC)
      # print(thresh)
      thresh = thresh.astype("float32") / 255.0
      thresh = np.expand_dims(thresh, axis=-1)
      thresh = thresh.reshape(1,32,32,1)
      if model_num==1:
          [y_pred] = model.predict(thresh)
          y_pred = y_pred.argmax()
  #         print(y_pred)
          pred = classes1[y_pred]
          # print(pred, y_pred)
          letters.append(pred)
      else:
        [y_pred2] = model2.predict(thresh)
        y_pred2 = y_pred2.argmax()
#         print(y_pred)
        pred2 = classes2[y_pred2]
        # print(pred2, y_pred2)
        letters.append(pred2)
      # letters.append(thresh)

    all_letters.append("".join(letters))
  
  all_letters_f = ""
  for w in all_letters:
    all_letters_f += (w + " ")
  return all_letters_f
      


@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        file1_type = file1.content_type
        if file1_type != 'image/jpeg' and file1_type != 'image/png':
            return 'Please upload image!'

        global file_name
        file_name = "orig."+file1.content_type.split('/')[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file1.save(file_path)
        words, image = get_words_ocr(file_path)
        cv2.imwrite("static/result.jpg", image)
        all_letters = get_letters_ocr(words)
        global final_result
        final_result = ''
        final_result = all_letters
        return redirect(url_for('result'))
    
    return render_template('ocr.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html', img_name="result.jpg", result_text=final_result)

if __name__ == '__main__':
    app.run()