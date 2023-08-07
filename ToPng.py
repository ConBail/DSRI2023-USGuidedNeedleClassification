import zipfile
import os
import glob
import nrrd
import numpy as np
from PIL import Image
import shutil
import random
import decimal

directory = "/home/conbail/projects/def-holden/usneedle_data/NoviceData"
directory_unzip_oop = "/home/conbail/scratch/Images/OOP"
directory_unzip_ip = "/home/conbail/scratch/Images/IP"
directory_data = "/Data/Ultrasound_Ultrasoun-Sequence.seq.nrrd"

def get_folder_from_path(path):
	if "oop" in path:
        
		new_path = path.replace("/home/conbail/scratch/Images/unzip_oop/", "")
	else:
		new_path = path.replace("/home/conbail/scratch/Images/unzip_ip/", "")
    
	return new_path.replace("/Data/Ultrasound_Ultrasoun-Sequence.seq.nrrd", "")
def nrrd_to_array(oop, ip):
	np_oop_videos={}
	np_ip_videos={}
	for path in oop:
		video, header = nrrd.read(path)
		np_oop_videos[get_folder_from_path(path)] = video
	for path in ip:
		video, header = nrrd.read(path)
		np_ip_videos[get_folder_from_path(path)] = video
        
        
	return np_oop_videos, np_ip_videos

def numpy_to_png(oop, ip):
	frame_counter = 0
	for key in oop.keys():
		frame_counter = 0
		selection = random.uniform(0,10)
		if (selection <=6.0):
			for frame in oop[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Train/oop/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1
		elif (selection <= 8.0):
			for frame in oop[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Val/oop/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1
		else:
			for frame in oop[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Test/oop/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1
	frame_counter=0
	for key in ip.keys():
		frame_counter = 0
		selection = random.uniform(0,10)
		if (selection <=6.0):
			for frame in ip[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Train/ip/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1
		elif (selection <= 8.0):
			for frame in ip[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Val/ip/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1
		else:
			for frame in ip[key]:
				img = Image.fromarray(np.squeeze(frame,axis=2))
				img.save("/home/conbail/scratch/PNG/Test/ip/"+str(frame_counter)+"_"+key+".png")
				frame_counter+=1

for filename in os.listdir(directory):
	try:
		number = filename.split("_")[0]
		if int(number)%2 == 0: # out of plane
			with zipfile.ZipFile("/home/conbail/projects/def-holden/usneedle_data/NoviceData/"+filename, 'r') as zip_ref:
				zip_ref.extractall("/home/conbail/scratch/Images/unzip_oop/")
		elif int(number)%2 !=0: # in-plane
			with zipfile.ZipFile("/home/conbail/projects/def-holden/usneedle_data/NoviceData/"+filename, 'r') as zip_ref:
				zip_ref.extractall("/home/conbail/scratch/Images/unzip_ip/")
	except:
		pass			


# Find sequence nrrd files for out of plane
oop_nrrd = glob.glob("/home/conbail/scratch/Images/unzip_oop/**/Ultrasound_Ultrasoun-Sequence.seq.nrrd", recursive=True)
ip_nrrd = glob.glob("/home/conbail/scratch/Images/unzip_ip/**/Ultrasound_Ultrasoun-Sequence.seq.nrrd", recursive=True)

folder_names = [get_folder_from_path(path) for path in oop_nrrd+ip_nrrd]


np_oop_videos, np_ip_videos = nrrd_to_array(oop_nrrd, ip_nrrd)
numpy_to_png(np_oop_videos, np_ip_videos)
