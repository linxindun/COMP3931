import os
import cv2
import numpy as np
import Extraction_labels
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    
    target_size=(224, 224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.
    return image

images_path = "RF\RFdataset\JPEGImages"
annotations_path = "RF\RFdataset/Annotations"

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
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predict_s = rf.predict(X_test)
accuracy = accuracy_score(y_test, predict_s)
print(f"Accuracy: {accuracy:.2f}")

tools_labels = ("Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag")
report = classification_report(y_test, predict_s, target_names=tools_labels)
print(report)
