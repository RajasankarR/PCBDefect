from flask import Flask
from DefectIdentification_lib import image_preprocessing,subtract_images,extract_contours_from_image,classify_defects
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import json
import uuid,os
app = Flask(__name__)


@app.route("/")
def hello():
    try:
        

        #load images
        print('started')
        image_path_1 = r"images\light_01_short_14_2_600.jpg"
        image_path_2 = r"images\light_01_short_14_2_600 - Copy.jpg"
        image_path_1 = r"images\l_light_01_short_13_3_600.jpg"
        image_path_2 = r"images\l_light_01_short_13_3_600 - Copy.jpg"
        Azure_connection_string='DefaultEndpointsProtocol=https;AccountName=pcbdefectdata;AccountKey=/heRERHi9/KddKIXuTJBZYbqg9amPfu3Q3EbEIr39vubseXkzKp5EYsYTagQ9N+b/Cxe4Ko57GOaNaNC1hQyIA==;EndpointSuffix=core.windows.net'
        processed_image = image_preprocessing(image_path_1, image_path_2)
        dissimilarity_path = subtract_images(processed_image[0], processed_image[1])
        defected_areas = extract_contours_from_image(**{
            "image_path": dissimilarity_path,
            "hsv_lower": [0, 150, 50],
            "hsv_upper": [10, 255, 255]
        })

        defect_types,filename = classify_defects(image_path_2, defected_areas)  # -> original_test_image, defect_dict
        print(filename)
        #Create a local directory to hold blob data
        local_path = "./output"

        #os.mkdir(local_path)
        Azure_connection_string = 'DefaultEndpointsProtocol=https;AccountName=pcbdefectdata;AccountKey=a095vsyfngGe6XR7LpFn31o5sezJnxy2C+g3S+OQmmsXwVXNBpyVTbVGC7NQWHdzstk73TaXZ9g43vrcDdkzyQ==;EndpointSuffix=core.windows.net'
        local_file_name=filename
        localfilepath=os.path.join(local_path,local_file_name)
        # Create a file in the local data directory to upload and download
        #local_file_name = str(uuid.uuid4()) + ".txt"
        upload_file_path = os.path.join(local_path, local_file_name)
        # Write text to the file
        file = open(upload_file_path, 'w')
        file.write("Hello, World!")
        file.close()
        container_name='pcbimagefolder'
        # Create a blob client using the local file name as the name for the blob
        blob_service_client = BlobServiceClient.from_connection_string(Azure_connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
        print("\nUploading to Azure Storage as blob:\n\t" + localfilepath)
        #upload_file_path=r'./pcbimagefolder/l_light_01_short_12_1_600.jpg'
        # Upload the created file
        with open(localfilepath, "rb") as data:
            blob_client.upload_blob(data,overwrite=True)



        return {'status':200,'containername':container_name,'filename':filename}
    except Exception as e:
        print(e)



if __name__=='__main__':
    app.run(debug=True)