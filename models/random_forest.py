import numpy as np
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score
from utils.write import write

if __name__ == '__main__':
    start_time = time()

    train_images = np.load('..//numpy_data/train_data.npy')
    train_labels = np.load('../numpy_data/train_labels.npy')
    validation_images = np.load('../numpy_data/validation_data.npy')
    validation_labels = np.load('../numpy_data/validation_labels.npy')
    test_images = np.load('../numpy_data/test_data.npy')

    label_scaler = preprocessing.LabelEncoder()
    label_scaler.fit(train_labels)
    train_labels = label_scaler.transform(train_labels)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_images)
    train_images = scaler.transform(train_images)
    validation_images = scaler.transform(validation_images)
    test_images = scaler.transform(test_images)

    model = RandomForestClassifier()
    model.fit(train_images, train_labels)

    print(f1_score(validation_labels, model.predict(validation_images), average='binary'))

    predicted_labels = model.predict(test_images)
    write(predicted_labels)

    end_time = time() - start_time
    print(f"------------------- {end_time} seconds -------------------")

# 0.5545927209705372 cu slice 7000 fara efecte
# 0.5614617940199335 cu slice 6000 fara efecte
