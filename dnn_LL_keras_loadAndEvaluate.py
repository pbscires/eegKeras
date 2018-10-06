import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Load the saved model and try to evaluate on new data
# load json and create model
json_file = open('model_tmp.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_tmp.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
# filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LL_PreIctal/chb04.csv'
dataset = np.loadtxt(filePath, delimiter=',')
X = dataset[:,:19]
y = dataset[:,19]

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))