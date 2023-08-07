import numpy as np
import nrrd
import tensorflow as tf
from keras.callbacks import CSVLogger
import tensorflow_addons as tfa
training_set = tf.keras.utils.image_dataset_from_directory(
	"/home/conbail/scratch/PNG/Train",
	labels="inferred",
	label_mode="binary",
	class_names=None,
	color_mode="rgb",
	batch_size=32,
	image_size=(224, 224),
	shuffle=True,
	seed=42,
	validation_split=None,
	subset=None,
	interpolation="bilinear",
	follow_links=False,
	crop_to_aspect_ratio=True,
)
val_set = tf.keras.utils.image_dataset_from_directory(
        "/home/conbail/scratch/PNG/Val",
        labels="inferred",
        label_mode="binary",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=42,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=True,
)


model_eff = tf.keras.models.load_model('efficient_base.h5')
model = tf.keras.Sequential([
	model_eff,
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dense(64, activation="relu"),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")])


precision = tf.keras.metrics.Precision(name="precision")
recall = tf.keras.metrics.Recall(name="recall")

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy",precision,recall])



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath="/home/conbail/projects/def-holden/conbail/UltrasoundApproachClassification/efficient_trained_more_metrics_long_test.h5",
	save_weights_only=False,
	monitor='loss',
	mode='min',
	save_best_only=True,
	save_freq='epoch')
csv_logger = CSVLogger('training_long.csv',append=False)

model.fit(training_set,validation_data=val_set, epochs=6,callbacks = [csv_logger,model_checkpoint_callback]) #Test on validation set after every epoch, and stop when results plateau



