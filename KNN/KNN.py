import os
import cv2
import numpy as np
import Extraction_labels
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    
    target_size=(224, 224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.
    return image

images_path = "KNN\KNNdataset\JPEGImages"
annotations_path = "KNN\KNNdataset/Annotations"

image_files = sorted(os.listdir(images_path))
annotation_files = sorted(os.listdir(annotations_path))
images = []
labels = []
for img, ann in zip(image_files, annotation_files):
    image_path = os.path.join(images_path, img)
    annotation_path = os.path.join(annotations_path, ann)
    image = load_image(image_path)
    labelslist = Extraction_labels.Extraction(annotation_path)
    preprocessed_image = preprocess_image(image)
    labels.append(labelslist[0])
    images.append(preprocessed_image)

tools_class = list(set(labels))
label_num = {label: num for num, label in enumerate(tools_class)}
all_labels = [label_num[label] for label in labels]

X = [img.flatten() for img in images]
X = np.array(X)
Y = np.array(all_labels) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
k_values = list(range(1, 101))
param_dist = {'n_neighbors': k_values}

random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

k_value = random_search.best_params_['n_neighbors']
score = random_search.best_score_

print("k:", k_value)
print("score:", score)

best_knn = KNeighborsClassifier(n_neighbors=k_value, metric='euclidean')
best_knn.fit(X_train, y_train)
predict_s = best_knn.predict(X_test)


accuracy = accuracy_score(y_test, predict_s)
print(f"Accuracy: {accuracy:.2f}")

tools_labels = ("Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag")
report = classification_report(y_test, predict_s, target_names=tools_labels)
print(report)
