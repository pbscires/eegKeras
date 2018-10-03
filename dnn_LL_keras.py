import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=19, activation='relu')) 
    model.add(Dense(8, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
filePath = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength/LineLength.chb03_01.edf.csv'
dataset = np.loadtxt(filePath, delimiter=',')
X = dataset[:,:19]
y = dataset[:,19]

# checkpoint
chkp_filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# create model
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0, callbacks=callbacks_list)
# evaluate using 5-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
