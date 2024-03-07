# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Put some global variables (e.g. path, etc.) here.

import os
#############################################################################
# Path for downloading the raw PDFs
raw_data_path = "data/raw_data" # The path to the raw PDFs
list_path = "data/target_list" # The path to the target list
tcfd_list = os.path.join(list_path, "tcfd.csv") # The path to the original TCFD list
esg_list = os.path.join(list_path, "esg.csv") # The path to the original ESG list
updated_tcfd_list = os.path.join(list_path, "tcfd_new.csv") # The path to the updated TCFD list
updated_esg_list = os.path.join(list_path, "esg_new.csv") # The path to the updated ESG list

#############################################################################
# Path for examples
example_path = os.path.join("data/examples")
# Path for the extracted pdf & text
extract_path = os.path.join("data/examples/extractText_sus", "")

#############################################################################
# Path for financial instruction tuning dataset
cache_dir = os.path.join("data/", "fin_instruction")