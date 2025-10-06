Repository for the paper "Using a Stack to Find an AI Needle: Topic Modeling for Cyber Threat Intelligence", currently submitted to ACM DTRAP.

We will publicly release this repository after acceptance of our paper.

## Description


* ```questionnaire.pdf``` contains the questionnaire distributed to the CTI practitioners that participated in our user-study.
* ```forum-topic-model.py``` is sample code to run our method.
* ```inquiry_ChatGPT```, which is a folder containing the discussion with ChatGPT (refer to Section 6.2 of the paper). [UPDATE: due to changes to OpenAI, we have upladed a new file, ``inquiry_ChatGPT-loaded.html`` with the correct rendering; the previous file has been renamed to "inquiry_ChatGPT-source_old.html"]. To see our discussion, download the html file and open it with any browser.

## Using the Topic Model Code

This code is meant to be run in batch mode using a GPU that supports PyTorch.

1. Create a python virtual environment
2. Install/verify the following packages (or run 'pip install -r mltopic-requirement.txt`):
	* contextualized_topic_models
	* pandas
	* wordcloud
	* matplotlib.pyplot
	* numpy
	* collections
	* seaborn
	* pathlib
	* warnings
	* os	
	* pickle
	* scipy

3. Activate virtual environment
4. Set the DATASET locations, encodings, and output directories
5. Run the code. The results will be written to the respective output directories

## Citing this work
If you use this code in your work, please cite our paper:

<formatted version goes here>

<bibtex version goes here>
