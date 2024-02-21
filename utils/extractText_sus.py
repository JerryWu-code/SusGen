import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from os import makedirs, path
import re
import fitz # PyMuPDF
from pandas import DataFrame
# from ftlangdetect import detect
from langdetect import detect
from collections import defaultdict
from os import path, scandir, listdir
from src.config import *

# Extract the text from the PDF document
# INPUT: PDF files 
# OUTPUT: A textual file for each PDF (.pdf --> .txt)

def extract_text(doc) -> str:
    
    # Iterate over the pages of the document
    documentText = ''
    for page in doc:

        # Get the text from the page
        pageBlocks = page.get_text('blocks', sort = False, flags = fitz.TEXTFLAGS_SEARCH & fitz.TEXT_DEHYPHENATE & ~ fitz.TEXT_PRESERVE_IMAGES)
        
        pageText = ''
        for block in pageBlocks:
            blockText = block[4] # STRUCTURE: (x0, y0, x1, y1, text, block_no, block_type)

            # a) Remove starting and ending whitespaces
            blockText = blockText.replace('\n', ' ').strip()
            pageText += blockText + '\n'

        # OTHER FUNCTIONS:  .get_links() // .annots() // .widgets():

        # b) Add full stop in different "sentences" --> e.g., improved readability \n In 2021 the company will ..
        pageText = re.sub(pattern = r'([a-zA-Z0-9 ])(\n+)([A-Z])', repl = r'\1. \2\3', string = pageText)
        
        # b) Remove the new line in the middle of a sentence --> e.g., improved readability \n of the code
        pageText = re.sub(pattern = r'([^.])(\n)([^A-Z])', repl = r'\1 \3', string = pageText)
            
        # c) Remove duplicated white spaces --> e.g., improved  readability
        pageText = re.sub(pattern = r' +', repl = ' ', string = pageText)

        # Save the page text in the document text --> pages separted by two new lines 
        documentText += pageText + '\n\n'
    
    # d) Remove duplicated page separators
    documentText = re.sub(pattern = r'\n{3,}', repl = r'\n\n\n', string = documentText)
    
    return documentText.strip()
    

def text_loader(report_data, analyze_language = True):

    for companyName, reports in report_data.items():
        print('\nCOMPANY:', companyName)
        
        for idk, url_report in enumerate(reports):

            # Read the PDF file
            try:
                with fitz.open(url_report['path']) as doc:    # type: ignore     
                    
                    document_text = extract_text(doc)
                    
                    # Ensure a good utf-8 encoding
                    document_text = document_text.encode('utf-8', errors = 'replace').decode('utf-8')
                    
                    # Check if the text is duplicated
                    is_duplicatedText = any([doc['text'] == document_text for doc in report_data[companyName] if 'text' in doc.keys() and doc['text'] != None])
                    if is_duplicatedText:
                        print("Duplicate text")
                        continue
                    
                    # Save the number of pages
                    report_data[companyName][idk]['numPages'] = doc.page_count           
                    
                    # Extract the text from the document
                    report_data[companyName][idk]['text'] = document_text
                    
                    # Extract the language
                    if analyze_language:
                        # predicted_language = detect(text = document_text.replace("\n"," "),  low_memory = False)
                        # report_data[companyName][idk]['language'] = predicted_language['lang']

                        predicted_language = detect(text=document_text.replace("\n", " "))
                        print("predicted_language:", predicted_language)
                        report_data[companyName][idk]['language'] = predicted_language
                    
                    print(f'--> REPORT {idk +1}/{len(reports)}: ' \
                        f"[{doc.metadata['format']}| PAGES:{doc.page_count}] '{url_report['documentType']}'")

            except RuntimeError as runtimeError:
                print(f'\t--> ERROR: {runtimeError}')
                report_data[companyName][idk]['text'] = None
                
    return report_data


def save_textualData(report_data, saving_folder):
    for companyName, reports in report_data.items():
        for report in reports:

            if 'text' not in report.keys() or report['text'] == None:
                continue

            # Create the folder for the year
            year_folder = path.join(saving_folder, report['year'])

            if not path.exists(year_folder):
                makedirs(year_folder)
                
            # File name 
            if 'language' in report.keys():
                fileName = companyName + "-" + report["documentType"] + "-" + report['language'].upper() + ".txt"
            else:
                fileName = companyName + "-" + report["documentType"] + ".txt"

            # Save the textual data
            with open(path.join(year_folder, fileName), mode = 'w', encoding = 'utf-8') as txt_file:
                
                try:
                    txt_file.write(report['text'])
                except UnicodeEncodeError as unicodeError:
                    print(f'\t--> ERROR: {unicodeError}')
                    report['text'] = report['text'].encode('utf-8', errors = 'replace').decode('utf-8')
                    txt_file.write(report['text'])
                    
def numPages_stats(saving_folder, report_data):
    df = DataFrame([report['numPages'] for reports in report_data.values() for report in reports 
                    if 'numPages' in report.keys()], columns=['numPages'])

    df = df.describe()    
    df.to_excel(path.join(saving_folder, 'numPages.xlsx'))
    
    return df

def documentMetadata_loader(folderPath):
    reports = defaultdict(list)
    for yearlyFolder in scandir(folderPath):
        print("yearlyFolder:", yearlyFolder.name)
        if not yearlyFolder.is_dir():
            print('A file was found in the folder', folderPath, 'and it was ignored')
            continue

        # Extract information
        year = yearlyFolder.name
        files = listdir(yearlyFolder.path)
        for fileName in files:
   
            #fileParts = fileName.split('.')
            #documentName = fileParts[0].strip() if len(fileParts) <= 2 else ''.join(fileParts[:-1])
            #fileExtension = fileParts[-1].strip()
            documentName, fileExtension  = path.splitext(fileName)

            if fileExtension.lstrip('.') not in ['pdf', 'txt', 'html', 'xhtml']:
                print('\nWARNING! Wrong extension for file "', fileName, '" in folder', yearlyFolder.path, 'and it was ignored')
                continue

            # Extract company name and document type
            partialComponents = documentName.split('-')
            
            if len(partialComponents) < 2 or len(partialComponents) > 3:
                raise Exception("\n" + documentName + "-->" + str(partialComponents) + '\nThe file name <<'+ fileName + '>> is not in the correct format!<companyName>-<documentType>[-<documentLanguage>]')

            # Parse the company name and document type
            partialComponents = [re.sub(pattern = r' +', repl = ' ', string = component.strip()) for component in partialComponents]
            companyName = partialComponents[0]
            documentType = partialComponents[1]
            
            # Parse the document language
            if len(partialComponents) == 3:
                documentLanguage = partialComponents[2]
            else:
                documentLanguage = None

            # Save information
            reports[companyName].append({
                'year': year,
                'documentType': documentType,
                'documentLanguage': documentLanguage,
                'fileExtension': fileExtension,
                'path' : path.join(yearlyFolder.path, fileName)})

    reports = dict(sorted(reports.items(), key = lambda dict_item: dict_item[0]))

    return reports
                   

if __name__ == '__main__':
    # Folder paths for the reports
    data_path = extract_path
    rawData_path = os.path.join(data_path, "")
    
    # Load the paths of the reports
    report_data = documentMetadata_loader(rawData_path)

    # Load the textual data
    report_data = text_loader(report_data)
    print(report_data)

    # Save the textual data (i.e., extracted texts)
    saving_folder = path.join(data_path, 'processed', '_REGEX')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    save_textualData(report_data, saving_folder)
    
    # Saving the stats
    save_numPages = numPages_stats(path.join(data_path, 'processed'), report_data)