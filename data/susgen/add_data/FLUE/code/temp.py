# import requests

# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
# filename = "fiqa.zip"

# response = requests.get(url)

# if response.status_code == 200:
#     with open(filename, 'wb') as f:
#         f.write(response.content)
#     print("File downloaded successfully.")
# else:
#     print("Failed to download the file. Status code:", response.status_code)

import zipfile

zip_file_path = "fiqa.zip"
extract_to_directory = "unzip"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_directory)

print("decompressed:", extract_to_directory)
