import zipfile
import os
import glob
import nrrd
import numpy as np
from PIL import Image
import shutil
import sys

def efficientNetV2_US_Classification(file_name, model_directory):

	working_dir = os.getcwd()

	os.mkdir(file_name + "_unzipped")
	os.mkdir("US_PNGs")

	try:
		with zipfile.ZipFile(working_dir + "/" + file_name, "r") as zip_ref:
			zip_ref.extractall(working_dir + "/" + file_name + "_unzipped")

	except:
		print("Something went wrong. Verify that you are following the instructions mentioned at the beginning of the program and try again.")
		sys.exit()


	nrrd_file = glob.glob(working_dir + "/" + file_name + "_unzipped/**/Ultrasound_Ultrasoun-Sequence.seq.nrrd", recursive=True)

	array_data, header = nrrd.read(nrrd_file[0])
	frame_counter = 0
	for frame in array_data:
		img = Image.fromarray(np.squeeze(frame, axis=2))
		img.save(working_dir + "/US_PNGs/" + str(frame_counter) + file_name + ".png")
		frame_counter += 1


	images = tf.keras.utils.image_dataset_from_directory(
        	working_dir + "/US_PNGs/" ,
        	labels=None,
        	label_mode=None,
        	class_names=None,
        	color_mode="rgb",
        	batch_size=32,
        	image_size=(224, 224),
        	shuffle=False,
        	seed=42,
        	validation_split=None,
        	subset=None,
        	interpolation="bilinear",
        	follow_links=False,
        	crop_to_aspect_ratio=True,
	)

	try:
		model = tf.keras.models.load_model(model_directory)
	except:
		print("Something went wrong. Please make sure you are passing the directory to the model as a string as the second command line argument.")

	preds = model.predict(images)


	ip_count = 0
	oop_count = 0
	for prediction in preds:
		print(prediction)
		if prediction == 0:
			ip_count +=1
		else:
			oop_count +=1
	if (ip_count>oop_count):
		print("The model predicts that the video is in-plane.")
	elif (oop_count>ip_count):
		print("The model predicts that the video is out-of-plane.)"
	else:
		print("The model is unsure.")

def main():
	file_name = sys.argv[1]
	model_directory = sys.argv[2]
	efficientNetV2_US_Classification(file_name,model_directory)

main()
