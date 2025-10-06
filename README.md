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
4. Set the following parameters: DATASET locations, encodings, and output directories
   - DATASET Locations: Full path to all datasets that will be used for both training and modeling:
     ```
	 local_datasets = {
		'DATASETNAME': '/path/to/dataset.csv',
		'STACK' : '/path/to/stackexchange.csv',
		'KAGGLE' : '/path/to/kaggle.csv',
		}
     ```
   - List of stopwords that should be ignored during text modeling:
     ```
     STOPWORD_LIST = '/path/to/StopWords.txt'
     ```
   - Output Directory: Output data will be saved in this base directory, within multiple subdirectories: 
     ```
     MODEL_SAVE_PATH = '/path/to/directory/Data/'
     ```	
   - Encoding: The text encoding used for each dataset:
	 ```
	 text_codec = { 
		'DATASETNAME': 'encoding',
		'STACK' : 'encoding',
		'KAGGLE' : 'encoding',
		}
   	  ```

6. Run the code. The results will be written to the respective output directories

## Citing this work
If you use this code in your work, please cite our paper:

Saskia Laura Schröer, Jeremy D. Seideman, Shoufu Luo, Giovanni Apruzzese, Sven Dietrich, and Pavel Laskov. 2025. Using a Stack to Find an AI Needle: Topic Modeling for Cyber Threat Intelligence. Digital Threats Just Accepted (September 2025). https://doi.org/10.1145/3766908

	
BibTeX:
```
@article{10.1145/3766908,
author = {Schr\"{o}er, Saskia Laura and Seideman, Jeremy D. and Luo, Shoufu and Apruzzese, Giovanni and Dietrich, Sven and Laskov, Pavel},
title = {Using a Stack to Find an AI Needle: Topic Modeling for Cyber Threat Intelligence},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3766908},
doi = {10.1145/3766908},
abstract = {Cyber Threat Intelligence (CTI) is a fundamental activity to ensure the protection of modern organizations against sophisticated cyberattackers. A large body of literature has addressed problems related to CTI. Despite the scientific validity of such results, the reality is that CTI practitioners rarely deploy advanced CTI methods proposed by the research community and mostly rely on manual processes. We seek to facilitate the manual analyses typical for CTI practice by proposing a novel topic modeling technique that enables analysts to identify specific topics in CTI data sources. We demonstrate how our method, released as an open-source tool, can be used to investigate three case studies revolving around the research question whether attackers are deploying AI for malicious purposes “in the wild,” and, if so, what features of AI interest them the most. We analyzed 7 million discussions from 18 underground forums. Our findings reveal that attackers may favor easy-to-use AI toolkits over the sophisticated AI techniques envisioned in research papers. Our contributions are further validated by a user study (N=24) with CTI experts, confirming the relevance of our research. Ultimately, we advocate future endeavors to account for the opinion of CTI practitioners—who should, in turn, try to cooperate.},
note = {Just Accepted},
journal = {Digital Threats},
month = sep,
keywords = {Cyber Threat Intelligence, Underground Forums, User Study}
}
```


