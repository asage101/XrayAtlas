# XrayAtlas
XrayAtlas is an automated X-ray classification and registration system designed to support medical imaging workflows. Its goal is to reduce human error and improve efficiency in healthcare, while also building a foundation for future machine learning applications.



## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/Ee6kO7cinINLooeKdn8ViQkB3oTuvH0Tr2U7wwR0N1VgxA?e=3B3SqY) | [Slack channel](https://cmpt419spring2025.slack.com/archives/C086RGK8TEV) | [Project report](https://www.overleaf.com/8591374339sfbpfqxfvjjj#9bc4b8) |
|-----------|---------------|-------------------------|



## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Project demo

[![YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://youtu.be/61XbSw7Ck1A)
[![XrayAtlas Demo](./images/XrayAtlas.png)](https://youtu.be/61XbSw7Ck1A)


### Project Structure


```bash
repository
├── src                          ## Source code
├── Classification.ipynb         ## Jupyter Notebook 
├── README.md                    ## Project Description
├── requirements.yml             ## Requirements to run the project
├── main.py                      ## Project entry
├── predict_image_here           ## Folder to add sample images for registration
```

<a name="installation"></a>

## 2. Installation


```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate xray_atlas
mkdir data
cd ./data
kaggle competitions download -c unifesp-x-ray-body-part-classifier
```

<a name="repro"></a>
## 3. Reproduction
Unzip the unifesp-x-ray-body-part-classifier compressed file, its contents should be in data/unifesp-x-ray-body-part-classifier/(content here)
```bash
python main.py --input filepath
Example is:
python .\main.py --input .\predict_image_here\sample.png
```
 
Data can be found at ./data after installation

Output will be displayed and model will be saved at main folder for .keras and at ./output for .h5

<a name="guide"></a>
## 4. Guidance

- Create a profile on [Kaggle](https://www.kaggle.com)
- Install Kaggle CLI depending on your [platform](https://www.kaggle.com/docs/api)
- Join the [competition](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier/data)
- Perform all the actions listed in Installation part
- Ensure that the paths for all the data folders are correct in main.py
- Run actions listed in Reproduction
