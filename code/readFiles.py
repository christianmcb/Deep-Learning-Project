import os

def ReadFiles(directory):
    for root, dirs, files in os.walk(directory):
        files_ = sorted([f'{directory}/{file}' for file in files], key=str)
        return files_