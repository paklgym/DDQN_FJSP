import os
import streamlit as st
    
def list_files():
    files = []
    path = "datasets"
    for entry in os.listdir(path):
        if os.path.isfile(os.path.join(path, entry)):
            files.append(entry)
    files.sort()
    return files

def read_file(filename):
    file = open(f'datasets/{filename}', "r")
    return file
