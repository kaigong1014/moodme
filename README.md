# Documentation

## What was done - Step by Step to create the app

1. Browsed through HuggingFace Hub to select candidate models and datasets for the project, for each task. See Section on models and datasets.
2. After selecting some good candidates, the finetuning step started with the first model - sentiment analysis. For this, Colab was used.
3. With the first finetuned model, started the creation of the app. A virtual environment was created as this was done locally. See setup section
4. The app UI was created to test the finetuned model and the other tasks, using models from the hub. No deployment on this step.
5. With the app UI ready, finetuning was done for the other two models, also using Colab. See Finetuning section
6. With all models ready, app code was modified to call the API for each finetuned model 
7. App was deployed. See deployment section


## About the models and datasets
The first step to create the app was to select the relevant models and datasets to use down the line. This was done by browsing through the HuggingFace Model and Dataset Hubs. When selecting the models and datasets, a lot of back and forth was done between testing the model's API, checking for good datasets to finetune and checking the model's and dataset sizes. For the finetuning step, the biggest risk was to run out of processing power to finetune the models. Therefore, it was necessary to select small models and small datasets or, datasets that could be subsetted without loss of quality. 

To choose datasets, the best approach is to look for trustworthy datasets. HuggingFace provides some datasets by standard, that are not in other user's repos. These datasets usually are obtained from trustworthy sources, where bias and mislabeling was not as common. Some of the datasets are used as benchmark for models, some of them were used in research studies, etc. This was the basis for selecting the datasets: the source and quality of the data.

For the models, each task can be done by using a different type of model. For sentiment analysis, encoder-only tranformer models are ideal, as this task require the model to understanding the input to be classified. Examples of encoder models are BERT, DistilBERT and RoBERTa. For text summarization a Encoder-decoder transformer model was necessary as these models are ideal for generative tasks that require an input such as a text to be summarized. Example of encoder-decoder models are BART and T5. Finally, to classify images to check if the person is using a mask or not, encoder-only models also can be used, like the ViT model.

At the end, the models used for each task were:
- Sentiment Analysis: distilbert/distilbert-base-uncased [https://huggingface.co/distilbert/distilbert-base-uncased]
- Text Summarization: google-t5/t5-small [https://huggingface.co/google-t5/t5-small]
- Image Classification: vit-base-patch16-224-in21k[https://huggingface.co/google/vit-base-patch16-224-in21k]

distilbert/distilbert-base-uncased was selected for the sentiment analysis task mainly due to its size. This model is smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, which means a human supervised the model training, which made the model reach higher accuracy. As the model card states "this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation you should look at model like GPT2."

google-t5/t5-small was used for the text summarization task for the same reason. Its also a smaller model that offers a lot of potential to be finetuned, since it was pretrained on many datasets for many different purposes. 

Finally, vit-base-patch16-224-in21k was used for image classification due to its pretrained version being already made for image classification, even tough there were smaller models.

The datasets used to finetune each model were:
- Sentiment Analysis: sst2
- Text Summarization: billsum
- Image Classification: face mask from kaggle - external dataset 

sst2 dataset, the Standr[.....] was the chosen dataset to finetune distilbert/distilbert-base-uncased model as it was a dataset provided by HuggingFace itself and used in the GLUE benchmark. The GLUE benchmark is "a collection of resources for training, evaluating, and analyzing natural language understanding systems".In summary, GLUE provides a standardized way to evaluate and compare NLP models. Therefore, this dataset was chosen for this task as it contained the necessary data.

billsum was chosen to finetune the google-t5/t5-small as it was also provided by HuggingFace. Another candidate for this finetuning was the CNN daily mail dataset but the billsum was chosen due to a more distributed lenght on inputs (most inputs were about similar size) and for being a smaller dataset. Ehile billsum had a around 23k rows, CNN dailymain presented 300k rows. Additionally, CNN dailymail contained more variable inputs lenghts and subsetting this dataset could cause problems on the model, in case only short or long inputs were used. To simplify the finetuning step and avoid adding too much preprocessing, a subset ready to use from the billsum was applied.

Finally, for the vit-base-patch16-224-in21k model, the face mask dataset was used mainly due to the task chosen. Since the goal was to create a model capable of detecting if a person on the image was using a mask, this dataset offered the right training data in the right format. Additionally, is was still trustworthy as it was a dataset used on a Kaggle competition : https://dphi.tech/challenges/data-sprint-76-human-activity-recognition/233/data


# Setup 
To finetune all models the Google Colab notebooks were used, with a hardware accelerator of t4 GPU. Google Colab offered 15GB o GPU memory.
To create the app, the streamlit framework was used. Streamlit is a Python framework used to simplify the creation of web interfaces. It also offers a free cloud service with a limitation of around ~2GB of storage. If the application deployed is smaller than this, the deployment and hosting of the app will be free of charge. 
To deploy the app, a github repo with the relevant code need to be created with a app.py script and a requirements.txt.

To initialize the project, a virtual enviroment was created with the following code:

```
python -m venv llm_app
llm_app\Scripts\activate
pip install <packages>
pip freeze > requirements.txt

```
This code was responsible for create the virtual environment, activating it, installing the relevant packages such as streamlit, transformers, [....] and creating the requirements.txt file necessary for deployment.


# Finetuning 
The first model that was finetuned was the distilbert/distilbert-base-uncased, for the sentiment analysis task. As mentioned, the sst2 datasetfrom HuggingFace itself, was used for this model. 

The finetuning was done on Colab and just a subset of the sst2 dataset, to test how long and how much computer power would be needed. Even with this config, the finetuning was taking a long time. The reason was that Colab was not utilizing GPUs, only CPU. After activating the GPU hardware accelerator on Colab, the finetuning took considerable less time, dropping from 2 hours to only 30 minutes with the whole dataset. After finetuning, the model final score was of 0.9014. This model was uploaded to the (hub)[https://huggingface.co/Kai1014/distilbert-finetuned]

The uploading step was necessary as this would allow calling the model directly on the streamlit app, without using much memory. If the model was saved as a pickle or any file format, to use it, it would be necessary to have the file itself either locally or on streamlit folder as a cached resource. This would take all memory from streamlit app and it would not be possible to deploy the app. Saving it into the hub would allow the app to call the model using an API.

With the first model uploaded to the hub, it was time to finetune the other two. The next model was the summarization model. Initially, the candidates selected were the "google/pegasus-xsum" and "facebook/bart-base", but as soon as the finetuning finetunin, using the billsum dataset, the limits of GPU resouces were reached on Colab. Therefore, the google-t5/t5-small was chosen as it was actually possible to train it using the Colab 15GB of GPU memory. This finetuning, however, was not done on the entire dataset as using the whole data would also extrapolate the Colab limitations. 

After finetuning, however, an issue was found during the testing step. It was observed that the T5 finetuned model was truncating the summarized outputs.
For example, for this input text, this was the summary obtained:

```
**Input:** The Eiffel Tower tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.

**Summary: , and the tallest structure in Paris. It is 324 metres (1,063
```

 After a long sets of tests and research on the HuggingFace community, the explanation obtained was that, by default, the `max_new_tokens` argument on the model.generate function that is called in the pipeline when accessing the model, limits this number to 20, and for this reason, the output was being truncated. Therefore, to obtain a good summary, when calling the model using pipeline the argument `max_new_tokens=300` was added in the app, in the testing fase after  finetuning of this model was complete. 
 
 The score metric used for this model was the ROUGE, or Recall-Oriented Understudy for Gisting Evaluation metric. This is a metric typically used to evaluate automatic summarization and machine translation software in NLP. The metric compared the produced summary to a reference produced by a human being. The ROUGE values are in the range of 0 to 1.

 The final model scores were:

"rouge1": unigram based scoring - 0.4154
"rouge2": bigram based scoring - 0.1753
"rougeL": Longest common subsequence based scoring - 0.2649
"rougeLSum": splits text using "\n" - 0.2649

These scores could be interpred as a percentage of the reference summary that are present in the machine-generated summary. For instance, the rouge1 score of 0.4154, indicates that 41% of the unigrams (words) on the reference summary are contained in the summary generated by the model.

The final model finetuned was the image detection model vit-base-patch16-224-in21k, finetuned with the face mask dataset from kaggle. Initially, however, the model DETR from facebook was the chosen model to be trained with the cppe5 dataset, to create a model capable of identifying medical protective equipment on the images. The model was finetuned but no predictions were being returned possibly due to the small size of the training data and epochs used to finetune, and due to the complexity of the training data, that contained multiple labels per image. Therefore, a smaller more speciallized model was chosen to be trained with a smaller more specialized dataset. The vit-base-patch16-224-in21k, after being trained on a dataset with mask, no_mask, mask_weared_incorrect labels was able to identify if the person on a picture was using a mask and using it properly. 
The finetuned version of the model reached an accuracy of 0.9716.

The challenge with finetuning this model with the facemask dataset was due to the terms of condition applied to the (original)[https://huggingface.co/datasets/poolrf2001/mask] dataset on HuggingFace. The dataset required authenthication and authorization to be accessed via the `load_dataset()` API and even after logging properly using the cli, `notebook.login()` and setting a secret within Colab, accessing the dataset was still retuning a 401 error of access denied. therefore, to contour the issue, it was decided to clone the repository of the dataset and upload it as a dataset repository in the personal account. 

For this, the code was used within Colab:

```
!git lfs install
!git clone https://huggingface.co/datasets/poolrf2001/mask
!huggingface-cli repo create mask-kaggle --type dataset

from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/content/mask",
    repo_id="Kai1014/facemask-kaggle",
    repo_type="dataset",
)
```
After this, the dataset could be accessed using the `load_dataset()` function as usual, and the finetuning was complete. 
from datasets import load_dataset

# Cicle of Development - CRISP-DM

Following the CRISP-DM process, the idea was to create a small version of the project with most of its functionality, as soon as possible, to find the possible issues right at the beggining. So, after finetuning the first model, the structure of the app, with all three tasks, was created. At this stage, only one of the models was already finetuned. The idea was to understand if the finetuned model performed as good as the others, how to call the models API on the app, and what was the format of the output, so it could be processed and displayed in a user friendly way.

This step helped in the understanding of future risks, and the need to upload the finetuned model into the Hub, to allow it to be called using the HuggingFace API. This would remove the need to load any object in the app itself that would make it impossible to deploy it and use later on, as it would exceed the storage limitations of the Streamlit framework. 

The first version of the app had the final UI (with smaller differences), but it was using the finetuned model for sentiment analysis, "cnicu/t5-small-booksum" model for summarization and the "nickmuchi/yolos-small-finetuned-masks" model for mask/no mask detection. These models were found during the initial phase when searching for good candidate models to continue the project. These models were finetuned by other people and performed a similar task as it was intended for the other models to be finetuned. 

After the intial finetuning and creation of the first version of the app, the next steps, following the cycle of development, was to finetune one of the models, modify the app's code, test the application and model's output and repeat the same steps for the last model. After all finetuning steps, the app was deployed and final tests were made in the deployed version.


# Step by step deployment
1. create a virtual enviroment, on an easy to access folder - python -m venv [name]
2. create another folder and unzip the folder there. Be aware that the items have to be directly on the folder you just created, not inside another folder.  
3. open visualstudio code or any code editor
4. access the app folder you just unzipped using the editor
5. go to the terminal on the editor and activate the venv [path]\Scripts\activate
6. install the packages using the requirements txt. With the venv activated type: python -m pip install -r requirements.txt on the terminal. The terminal have to be on the same path to the folder were the requirements is, otherwise, place the full path to the requirements file.
7. change the python interpreter on the app.py code to the one inside the venv. On visualstudio this is done by clicking on python on the side of the word 'Python', on the version, and selecting the interpreter on the global search bar. If you activated it, it would be on the recommended, top of the list.
8. create a github repo on the website, manually. 
9. push the app folder to a github repo. On the folder you created and unziped everything, using the terminal and making sure you are on the right path use these commands:
    git init
    git add .
    git commit -m "first commit"
    git branch -M main
    git remote add origin [url of the created repo]
    git push -u origin main
10. mount the app locally using streamlit run app.py on the terminal. This will launch the app locally. 
11. On the UI, on the top right, click on Deploy
12. Follow the steps deploying with github, giving it a url and wait for the app the be launched! 