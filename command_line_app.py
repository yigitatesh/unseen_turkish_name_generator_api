print("Loading Packages...")

import os

# Do not show unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# USE FULL POWER OF GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np


### Load Data
print("Loading Data...")

with open("data/names.txt") as f:
    names = f.readlines()


### Prepare Data

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

print("Loading the Generator...")
inference_model = tf.keras.models.load_model("model/tr_name_generate_inference_model_3.h5")


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

## user input functions
def get_seed_from_user():
    """Gets initial characters (seed) from user"""
    valid = False
    seed = "" #default value

    # user info
    print("\nType initial characters of name(s) to be generated.")
    print("(You can directly Enter without writing any characters)")

    while not valid:
        seed = input("Your input: ")

        if is_seed_valid(seed):
            seed = tr_upper_to_lower(seed)
            valid = True
        else:
            print("\nPlease type alphabetical character(s).\n")

    return seed

def get_num_names_from_user():
    """Gets number of names to be generated from user"""
    valid = False
    num_names = 1 #default value

    # user info
    print("\nType number of Turkish names to be generated.")
    print("(You can directly Enter to generate 1 name)")

    while not valid:
        num_names = input("Your input: ")

        if num_names.strip() == "":
            num_names = 1
            break

        try:
            num_names = max(int(num_names), 0)
            valid = True
        except:
            print("\nPlease type an integer number.\n")

    return num_names

# menu function
def menu():
    print("\nPress Enter to generate unseen turkish name(s).")
    print("Type q and press Enter to exit.")

## main function
def main():
    # dummy generation for initialization
    generate_unseen_names()

    print("\nWelcome to the Turkish Name Generator!")
    print("These created names will NOT be REAL NAMES!")
    print("They are being created by an Artificial Intelligence.")
    
    run = True
    while run:
        # print menu
        menu()

        # get input
        choice = input("\nYour input: ")

        # process choice
        if choice.strip() == "":
            ## generate names
            # get initial characters
            seed = get_seed_from_user()

            # get number of names
            num_names = get_num_names_from_user()

            # generate and print names
            print("\nYour Turkish Names:\n")

            generated_names = generate_unseen_names(seed=seed, num_names=num_names)
            for i, name in enumerate(generated_names):
                print("{}: {}".format(i+1, name))

        elif choice.lower().strip() == "q":
            # print goodby message and exit
            print("\nSee you again")
            run = False

        else:
            print("\nNot a valid choice!")


# START THE GENERATOR APP
if __name__ == "__main__":
    main()
