import numpy as np
import nrrd
import tensorflow as tf





def displayPredictions(ground_truth, pred):
	total_pred = 0
	correct_pred = 0
	for ground, pred in zip(ground_truth, pred):
		total_pred+=1
		if (pred>0.5):
			pred=1
		else:
			pred=0
		if (ground == pred):
			correct_pred+=1
			print("Ground: ", ground.numpy(), "|||| Prediction: ", pred, " |||| MATCH!")
		else:
			print("Ground: ", ground.numpy(), "|||| Prediction: ", pred)
	print("CORRECT: " , correct_pred)
	print("ACCURACY: ", ((correct_pred/total_pred)*100))



def getXY(dataset):
	X = []
	y = []
	counter = 0
	for batch, labels in dataset:
		for image in batch:
			X.append(image)
		for label in labels:
			y.append(label)
		if counter == 10:
			break
		counter +=1
	return X, y





test_set = tf.keras.utils.image_dataset_from_directory(
	"/home/conbail/scratch/PNG/Test",
	labels="inferred",
	label_mode="binary",
	class_names=None,
	color_mode="rgb",
	batch_size=32,
	image_size=(224, 224),
	shuffle=True,
	seed=42,
	validation_split=0,
	subset=None,
	interpolation="bilinear",
	follow_links=False,
	crop_to_aspect_ratio=True,
)

for batch, labels in test_set:
	for image in batch:
		image/255
	break

X_test, y_test = getXY(test_set)


X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)



model_final = tf.keras.models.load_model('efficient_trained.h5')

pred = model_final.predict(X_test)

displayPredictions(y_test, pred)
