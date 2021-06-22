import numpy as np
import librosa

from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import noisereduce as nr
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,request,Response
from flask import (Flask,g,redirect,render_template,request,session,url_for)


app = Flask(__name__,template_folder="templates",static_folder="static")
app.secret_key = 'check'


# @app.route('/login', methods=['GET', 'POST'])
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/profile.html')
def profile():

    return render_template('profile.html')

@app.route('/index.html')
def index():
    return render_template('index.html')



# Replicate label encoder
lb = LabelEncoder()
# label = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
#         'drilling', 'engine_idling', 'glassbreak', 'gun_shot',
#         'jackhammer', 'scream', 'siren', 'street_music']
label = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
       'engine_idling', 'glassbreak', 'gun_shot', 'scream', 'siren',
       'street_music']        
# label = ['dog_bark','gun_shot','glassbreak','scream']


    # model work
    # model = pickle.load(open('random.pkl', 'rb'))


    # Model reconstruction from JSON file
model_path = r"./"
model_name = "cnnfinal"
with open(model_path + model_name + '.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')


lb.fit_transform(label)

    
    


CHUNK = 2**11
RATE = 22050
selected_labels = ['gun_shot','dog_bark','scream','glassbreak','siren']


# @app.route('/')
# def home():
#     return render_template('index.html')




@app.route('/y_predict',methods=["GET","POST"])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    
    outlist=[]
    ots = ""
    output=""
    if(request.method == "POST"):
        print("FORM DATA RECEIVED")
        file = request.files["file"]
        # file.stream.seek(0) # seek to the beginning of file
        # myfile = file.file
        if("file" not in request.files):
            output="No File Uploaded"
            return redirect(request.url)

        
        if(file.filename == ""):
            output="No File Uploaded"
            return redirect(request.url)
        if(file):
            audio, sr = librosa.load(file,sr=22050)
            # dur = int(librosa.get_duration(y=audio,sr=sr))
            # Get number of samples for 2 seconds; replace 2 by any number
            buffer = 4 * sr

            samples_total = len(audio)
            samples_wrote = 0
            counter = 0
            rate = 22050
            a,b,c,d=0,0,0,0
            while samples_wrote < samples_total:
                #check if the buffer is not exceeding total samples 
                # if buffer > (samples_total - samples_wrote):
                #     buffer = samples_total - samples_wrote
                block = audio[samples_wrote : (samples_wrote + buffer)]
                block = np.fromstring(block, 'float32')
            #     out_filename = "split_" + str(counter) + "_" + file_name
                noise_len = 2 # seconds
                noise = band_limited_noise(min_freq=500, max_freq = 12000, samples=len(block), samplerate=sr)*10
                noise_clip = noise[:rate*noise_len]
                audio_clip_band_limited = block+noise
            #     noise_reduced = nr.reduce_noise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip,prop_decrease=1.0, verbose=False)
                noise_reduced = nr.reduce_noise(audio_clip=block, noise_clip=audio_clip_band_limited, prop_decrease=1.0,pad_clipping=True, use_tensorflow=True,verbose=False)
                mfccs_features = librosa.feature.melspectrogram(y=noise_reduced, sr=sr)
                mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
                
            #     print(mfccs_scaled_features)
                mfccs_scaled_features=mfccs_scaled_features.reshape(1,128,-1)
                
                samples_wrote += buffer
                probs = model.predict(mfccs_scaled_features)
                best_labels = np.argsort(probs[0])[:-4:-1]
                counter += 4
                chk1 = round(probs[0][best_labels[0]]*100) 
                chk2 = round(probs[0][best_labels[1]]*100) 
                chk3 = round(probs[0][best_labels[2]]*100) 
                if((label[best_labels[0]]=='glassbreak' and chk1>0) or (label[best_labels[1]]=='glassbreak' and chk2>0) or (label[best_labels[2]]=='glassbreak' and chk3>0)):
                    a+=1
                    
                if((label[best_labels[0]]=='gun_shot' and chk1>0) or (label[best_labels[1]]=='gun_shot' and chk2>0) or (label[best_labels[2]]=='gun_shot' and chk2>0)):
                    b+=1
                   
                if((label[best_labels[0]]=='dog_bark' and chk1>0) or (label[best_labels[1]]=='dog_bark' and chk2>0) or (label[best_labels[2]]=='dog_bark' and chk2>0)):
                    c+=1
                   
                if((label[best_labels[0]]=='scream' and chk1>90) or (label[best_labels[1]]=='scream' and chk2>90) or (label[best_labels[2]]=='scream' and chk3>90)):
                    d+=1
                   
                output += f'Predictions'
                for i in range(3):
                    chks = round(probs[0][best_labels[i]]*100)
                    labs= label[best_labels[i]]
                    if((labs=='gun_shot'or labs=='glassbreak' or labs=='dog_bark' or labs=='scream') and chks>0):
                        output = output + f'\n{label[best_labels[i]]} - {round(probs[0][best_labels[i]]*100)}% from {counter}s to {counter+4}s  \n' 
                 
                if(output!="Predictions"):
                    outlist.append(output) 
                    output=""
                else:
                    output="" 
            fin = [a,b,c,d]
            maxx = max(fin)
            fin1,fin2,fin3,fin4 = "","","",""
            if(a==0 and b==0 and c==0 and d==0):
                ots = f'Fortunately, No Suspicious sounds detected'
            else:    
                if(a>0):
                    fin1 = "GlassBreak "
                if(b>0):
                    fin2 = "Gunshot "
                if(c>0):
                    fin3 = "Dog Bark "
                if(d>0):
                    fin4 = "Scream "
                ots = fin1 + fin2 + fin3 + fin4

            res =f'{maxx}' 
            ots +=f'\n is detected !!!'


            
            
    return render_template('index.html',outlist=outlist,ots=ots)


if __name__ == "__main__":
    
    app.run()
