import numpy as np
import librosa
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import noisereduce as nr
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,request,Response
from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for
)

# login settings
# class User:
#     def __init__(self, id, username, password):
#         self.id = id
#         self.username = username
#         self.password = password

#     def __repr__(self):
#         return f'<User: {self.username}>'

# users = []
# users.append(User(id=1, username='raeespeer', password='password'))
# users.append(User(id=2, username='vigneshprabhu', password='password'))


app = Flask(__name__,template_folder="templates",static_folder="static")
app.secret_key = 'check'

# login process
# @app.before_request
# def before_request():
#     g.user = None

#     if 'user_id' in session:
#         user = [x for x in users if x.id == session['user_id']][0]
#         g.user = user
        

# @app.route('/login', methods=['GET', 'POST'])
@app.route('/')
def login():
  
    # try:
    #     if request.method == 'POST':
    #         # session.pop('user_id', None)

    #         # username = request.form['username']
    #         # password = request.form['password']
            
    #         # user = [x for x in users if x.username == username][0]
            
    #         # if user and user.password == password:
    #         #     session['user_id'] = user.id
                
    #             return redirect(url_for('profile'))

    #         # return redirect(url_for('login'))
    # except:
    #     return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/profile.html')
def profile():
    # if not g.user:
    #     return redirect(url_for('login'))

    return render_template('profile.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

model_path = r"./"
model_name = "cnnnew"
model=None
# Replicate label encoder
lb = LabelEncoder()
# label = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
#         'drilling', 'engine_idling', 'glassbreak', 'gun_shot',
#         'jackhammer', 'scream', 'siren', 'street_music']
label = ['dog_bark','gun_shot','glassbreak','scream']
def load_model():
    # model work
    # model = pickle.load(open('random.pkl', 'rb'))

    global model
    # Model reconstruction from JSON file
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
        if("file" not in request.files):
            output="No File Uploaded"
            return redirect(request.url)

        file = request.files["file"]
        
        if(file.filename == ""):
            output="No File Uploaded"
            return redirect(request.url)
        if(file):
            audio, sr = librosa.load(file)
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
                noise_len = 3 # seconds
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
                probs = model.predict(mfccs_scaled_features/128)
                best_labels = np.argsort(probs[0])[:-4:-1]
                counter += 1
                if(label[best_labels[0]]=='glassbreak' or label[best_labels[1]]=='glassbreak'):
                    a+=1
                    
                if(label[best_labels[0]]=='gun_shot' or label[best_labels[1]]=='gun_shot' or label[best_labels[2]]=='gun_shot'):
                    b+=1
                   
                if(label[best_labels[0]]=='dog_bark' or label[best_labels[1]]=='dog_bark'):
                    c+=1
                   
                if(label[best_labels[0]]=='scream' or label[best_labels[1]]=='scream'):
                    d+=1
                   
                output += f'Predictions'
                for i in range(3):
                    output = output + "\n" f'{label[best_labels[i]]} - {round(probs[0][best_labels[i]]*100)}% from {counter}s to {counter+4}s.\n' 
                            
                outlist.append(output) 
                output=""
            fin = [a,b,c,d]
            maxx = max(fin)
            fin1,fin2,fin3,fin4 = "","","",""
            if(a==0 and b==0 and c==0 and d==0):
                ots = f'Fortunately, No Suspicious sounds detected'
            else:    
                if(a>0):
                    fin1 = "GlassBreak "
                if(b > 0):
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
    load_model()
    app.run(debug=True)
