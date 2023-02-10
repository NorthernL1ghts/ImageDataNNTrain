import os
import csv

# specify the folder containing the images
folder = "cars_train"

# create an empty list to store the image filenames and labels
data = []

# iterate through all files in the folder
for filename in os.listdir(folder):
    # check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # add the filename and label (assume label is the same as the filename without the file extension) to the data list
        data.append([filename, filename.split(".")[0]])

# write the data to a CSV file
with open("labels.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Labels successfully written to labels.csv!")
