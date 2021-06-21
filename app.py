# Import Libraries
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, flash
from tensorflow.keras.models import load_model
import os

# Create Flask app
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

### Load Names Data

with open("data/names.txt", encoding="utf-8") as f:
    names = f.readlines()

### Get info from Names Data

# add start(<) and end(>) tokens
for i in range(len(names)):
    names[i] = "<" + names[i].strip() + ">"

## create dicts char to index and vice versa

# get all chars
all_chars_set = set("".join(names))
all_chars = sorted(all_chars_set)

vocab_size = len(all_chars) + 1 # +1 for empty character (padding)

# lookup dicts
char_to_index = dict([(ch, ix+1) for ix, ch in enumerate(all_chars)])
index_to_char = dict([(ix, ch) for ch, ix in char_to_index.items()])

# turkish characters upper to lower dict
tr_upper_to_lower_dict = {"Ç":"ç", "Ğ":"ğ", "İ":"i", "Ö":"ö", "Ş":"ş", "Ü":"ü", "I":"ı"}

# max sequence len
max_len = max([len(name) for name in names])

## Seq to Name and Name to Seq
def name_to_seq(name):
    return [char_to_index[ch] for ch in name]

def seq_to_name(seq):
    return "".join([index_to_char[i] for i in seq])


### Load the inference model

# Model Settings
embedding_dim = 64
# Load the Model
inference_model = load_model("model/tr_name_generate_inference_model_3.h5")


### Generate Names

## Helper Functions

def generate_name(seed):
    """Generate chars given a seed (it can be empty)
    
    In a loop, predicts next chars and appends them to seed
    Next chars are chosen using probabilities to create different names each time
    
    Parameters:
        seed (string):
            initial characters of the wanted name
    
    Returns:
        (string):
            generated name"""
    
    seed = "<" + seed
    output = seed
    
    # create initial states
    h_state = tf.zeros(shape=(1, embedding_dim))
    c_state = tf.zeros(shape=(1, embedding_dim))
    states = [h_state, c_state]
    
    stop = False
    
    while not stop:
        # convert text seed to model input
        seq = name_to_seq(seed)
        seq = np.array([seq])
        
        # predict next char
        probs, h_state, c_state = inference_model([seq] + states)
        states = [h_state, c_state]
        probs = np.asarray(probs)[:, -1, :]
        # 
        index = np.random.choice(list(range(vocab_size)), p=probs.ravel())
        
        if index == 0:
            break
            
        pred_char = index_to_char[index]
        seed = pred_char
        output += pred_char
        
        if pred_char == ">" or len(output) > 20:
            break
    
    return output[1:-1] # get rid of start(<) and end(>) chars

def is_real_name(name):
    """checks whether created name is in names dataset or not"""
    name = "<" + name.strip() + ">"
    for real_name in names:
        if name == real_name:
            return True
    return False

def generate_unseen_names(seed="", num_names=1):
    """Generates unseen turkish names
    
    Parameters:
        seed (string) = "":
            initial characters of the wanted name
        num_names (integer) = 1:
            number of names to be generated
    
    Returns:
        (list of strings):
            generated names
    """
    generated_names = []
    
    i = 0
    while i < num_names:
        # generate a name
        name = generate_name(seed=seed)
        
        # check whether name is in dataset or not
        if not is_real_name(name):
            generated_names.append(name)
            i += 1
    
    return generated_names

def is_seed_valid(seed):
    """Checks whether a given seed is valid or not"""
    for ch in seed:
        if not ch in all_chars_set:
            return False
    return True

def tr_upper_to_lower(text):
    """Converts a turkish text from uppercase to lowercase"""
    out = []
    for ch in text:
        if ch in tr_upper_to_lower_dict:
            out.append(tr_upper_to_lower_dict[ch])
        else:
            out.append(ch.lower())
    
    return "".join(out)


# Flask Functions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ## preprocess given features
    features = list(request.form.values())
    are_inputs_valid = True

    # process inital characters
    seed = features[0]
    if not is_seed_valid(seed):
        flash("Please type alphabetical character(s) as initial characters of names.")
        are_inputs_valid = False
    else:
        seed = tr_upper_to_lower(seed)

    # process number of names
    num_names = features[1]
    if num_names.strip() == "":
        num_names = 1
    else:
        try:
            num_names = max(int(num_names), 0)
        except:
            flash("Please type an integer number as number of names.")
            are_inputs_valid = False

    # return if inputs are invalid
    if not are_inputs_valid:
        return render_template("index.html")

    # generate names
    generated_names = generate_unseen_names(seed=seed, num_names=num_names)

    return render_template('index.html', generated_names=generated_names)


if __name__ == "__main__":
    app.run(debug=True)