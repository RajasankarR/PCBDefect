import cv2
import numpy as np
import os, time, uuid

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


ENDPOINT='https://southcentralus.api.cognitive.microsoft.com/'
prediction_key='9d5edb22c7da41da8d91d736e537c7c4'
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
Directory="C:\Analytics Projects\DefectDetectionProject"
confidences_threshold = 10

# Working Folders
working_dir = 'temp'
output_dir = 'output'

# create dirs
os.makedirs(working_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)



def subtract_images(image_path_1, image_path_2):
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    # Difference image pa  th
    (dirname, filename) = os.path.split(image_path_2)
    filename = filename.split(".")[0]+ "_diff." + filename.split(".")[1]
    write_path = os.path.join(working_dir, filename)

    # Find differences
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]
    cv2.imwrite(write_path, image1)

    return write_path

def extract_contours_from_image(image_path, hsv_lower, hsv_upper):
    filename = os.path.basename(image_path).split(".")[0]

    image = cv2.imread(image_path)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(hsv_lower)
    hsv_upper = np.array(hsv_upper)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 20
    ROI_number = 0
    extract_defects = {}
    defect = {}

    for c in cnts:
        defect_area = {'defect_image':'', 'defect_area':''}

        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (36, 255, 12), 2)
        ROI = original[y - offset:y + h + offset, x - offset:x + w + offset]
        try:
            resized = cv2.resize(ROI, (64, 64), interpolation = cv2.INTER_AREA)
            out_path = os.path.join(working_dir,'{}_{}.png'.format(filename, ROI_number))
            cv2.imwrite(out_path, resized)

            defect_area['defect_image'] = out_path
            defect_area['defect_area'] = [(x - offset, y - offset), (x + w + offset, y + h + offset)]
            defect[ROI_number] = defect_area

        except exception as e:
            print("skipping image " + image_path)
        ROI_number += 1


    extract_defects[filename] = defect
    # cv2.imwrite(diff_path, image)
    return extract_defects

def predict_defect(file):
    with open(file, "rb") as image_contents:
        results = predictor.classify_image('ddf22915-2f29-44a5-b8e7-2d0130a4dd78', 'working_model', image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        if prediction.probability * 100 < confidences_threshold:
            return ('NOT')
        return (prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))


def classify_defects(image_path, defects):
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    final_out = os.path.join(output_dir, filename)

    for key, value in defects.items():
        for ind, val in value.items():
            
            defect_image = defects[key][ind]['defect_image']
            img = cv2.imread(defect_image)
            defect_area = defects[key][ind]['defect_area']  # [(36, 19), (80, 62)]
            x_co = defect_area[0][0] - 5
            y_co = defect_area[0][1] - 5

            status = predict_defect(defect_image)

            if 'NOT' in status:
                continue
            cv2.putText(image, status, (x_co, y_co), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.rectangle(image, tuple(defect_area[0]), tuple(defect_area[1]), (0, 0, 255), 2)

        cv2.imwrite(final_out, image)
    return True,filename

def image_preprocessing(image_path_1, image_path_2):
    images = [image_path_1, image_path_2]
    out_path = []

    for img in images:
        (dirname, filename) = os.path.split(img)
        filename = filename.split(".")[0]+ "_BW." + filename.split(".")[1]
        write_path = os.path.join(working_dir, filename)
        originalImage = cv2.imread(img)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY_INV)
        
        cv2.imwrite(write_path, blackAndWhiteImage)
        out_path.append(write_path)

    return out_path


def fn_uploadfile_toBlob(local_file_name):
    local_path = "./output"
    # os.mkdir(local_path)
    Azure_connection_string = 'DefaultEndpointsProtocol=https;AccountName=pcbdefectdata;AccountKey=a095vsyfngGe6XR7LpFn31o5sezJnxy2C+g3S+OQmmsXwVXNBpyVTbVGC7NQWHdzstk73TaXZ9g43vrcDdkzyQ==;EndpointSuffix=core.windows.net'
    localfilepath = os.path.join(local_path, local_file_name)
    # Create a file in the local data directory to upload and download
    # local_file_name = str(uuid.uuid4()) + ".txt"
    upload_file_path = os.path.join(local_path, local_file_name)
    # Write text to the file
    file = open(upload_file_path, 'w')
    file.write("Hello, World!")
    file.close()
    container_name = 'pcbimagefolder'
    # Create a blob client using the local file name as the name for the blob
    try:

        blob_service_client = BlobServiceClient.from_connection_string(Azure_connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
        print("\nUploading to Azure Storage as blob:\n\t" + localfilepath)
        # upload_file_path=r'./pcbimagefolder/l_light_01_short_12_1_600.jpg'
        # Upload the created file
        with open(localfilepath, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        status=False
    status=True
    return status

if __name__=="__main__":

    processed_image =  image_preprocessing(image_path_1, image_path_2)
    dissimilarity_path = subtract_images(processed_image[0], processed_image[1])
    defected_areas = extract_contours_from_image(**{
            "image_path" : dissimilarity_path,
            "hsv_lower" : [0,150,50],
            "hsv_upper" : [10,255,255]
        })

    defect_types = classify_defects(image_path_2, defected_areas) # -> original_test_image, defect_dict

