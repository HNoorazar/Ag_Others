# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import requests
import pandas as pd

# %%
# url = 'https://www.facebook.com/favicon.ico'
# r = requests.get(url, allow_redirects=True)
# open('facebook.ico', 'wb').write(r.content)

# %%
# Change working directory if you want:

import os
print ("I am currently in " + os.getcwd())

data_dir = "/Users/hn/Documents/01_research_data/MODIS/"
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)
print ("I am currently in " + os.getcwd())

# %%
from bs4 import BeautifulSoup

# %%
href = "https://zenodo.org/record/files/"
ext = '.zip'

# %%

# %%
from urllib.request import Request, urlopen, urlretrieve
from bs4 import BeautifulSoup

# %%

# %%
href="https://zenodo.org/record/files/"
def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True

print (is_downloadable(href))

# %%
print (is_downloadable('http://google.com/favicon.ico'))

# %%

# %%

# %%

# %%

# %%

# href = 'http://cdimage.debian.org/debian-cd/8.2.0-live/i386/iso-hybrid/'
# ext = 'iso'

# href = "https://zenodo.org/record/files/"
# ext = '.zip'

# def read_url(url):
#     url = url.replace(" ","%20")
#     req = Request(url)
#     a = urlopen(req).read()
#     soup = BeautifulSoup(a, 'html.parser')
#     x = (soup.find_all('a'))
#     for i in x:
#         file_name = i.extract().get_text()
#         url_new = url + file_name
#         url_new = url_new.replace(" ","%20")
#         if(file_name[-1]=='/' and file_name[0]!='.'):
#             read_url(url_new)
#         print(url_new)

# read_url("https://zenodo.org/record/files/")

# def listFD(url, ext=''):
#     page = requests.get(url).text
#     print (page)
#     soup = BeautifulSoup(page, 'html.parser')
#     return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

# for file in listFD(href, ext):
#     print (file)
    
# def get_url_paths(url, ext='', params={}):
#     response = requests.get(url, params=params)
#     if response.ok:
#         response_text = response.text
#     else:
#         return response.raise_for_status()
#     soup = BeautifulSoup(response_text, 'html.parser')
#     parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
#     return parent

# url = 'http://cdimage.debian.org/debian-cd/8.2.0-live/i386/iso-hybrid/'
# ext = 'iso'

# url = "https://zenodo.org/record/files/"
# ext = '.zip'

# result = get_url_paths(url, ext)
# print(result)

# %%

# %%
firsts = ["07", "08", "09"]+[str(x) for x in list(range(10, 22))]
seconds = ["v0" + str(x) for x in range(1, 10)] + ["v"+str(x) for x in range(10, 15)]

# %%
counter=0
for firsr_part in firsts:
    for second_part in seconds:
        if (counter % 50 == 0):
            print ("counter is [{:.0f}].".format(counter))
        fileName = "h" + firsr_part + second_part + ".zip"
        file_path =  href + fileName
        # print (file_path)
        r = requests.get(file_path, allow_redirects=True)
        open(fileName, 'wb').write(r.content)
        counter+=1

# %%

# %%

# %%

# %%

# %%

# %%
