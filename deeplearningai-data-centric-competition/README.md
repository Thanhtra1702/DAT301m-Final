# deeplearningai-data-centric-competition

![title](https://github.com/Nov05/deeplearningai-data-centric-competition/blob/bb77e84c77bd41d1374ae32512b2bc08f10f7edb/images/Data-Centric%20AI%20Competition.jpg)  

## About the competition

https://https-deeplearning-ai.github.io/data-centric-comp/  

A collaboration between DeepLearning.AI and Landing AI, the Data-Centric AI Competition aims to elevate data-centric approaches to improving the performance of machine learning models. In most machine learning competitions, you are asked to build a high-performance model given a fixed dataset. However, machine learning has matured to the point that high-performance model architectures are widely available, while approaches to engineering datasets have lagged. The Data-Centric AI Competition inverts the traditional format and instead asks your to improve a dataset given a fixed model. We will provide you with a dataset to improve by applying data-centric techniques such as fixing incorrect labels, adding examples that represent edge cases, apply data augmentation, etc. Contestants must submit their altered dataset for evaluation by September 4, 2021. (We picked this date because it is the birthday of John McCarthy, who had coined the term artificial intelligence!) The top three winners from each of the two categories (Best Performance Overall and Most Innovative) will be invited to a private event with Andrew Ng to share ideas about how to grow the data-centric movement, and will be highlighted in The Batch and other DeepLearning.AI and Landing AI channels. Good luck!


## Data-Centric AI Competition Submission Guide

https://worksheets.codalab.org/worksheets/0x7a8721f11e61436e93ac8f76da83f0e6

### Introduction

A collaboration between DeepLearning.AI and Landing AI, the Data-Centric AI Competition aims to elevate data-centric approaches to improving the performance of machine learning models. In most machine learning competitions, you are asked to build a high-performance model given a fixed dataset. However, machine learning has matured to the point that high-performance model architectures are widely available, while approaches to engineering datasets have lagged. The Data-Centric AI Competition inverts the traditional format and instead asks you to improve a dataset given a fixed model. We will provide you with a dataset to improve by applying data-centric techniques such as fixing incorrect labels, adding examples that represent edge cases, apply data augmentation, etc. Contestants must submit their altered dataset for evaluation by September 4, 2021. (We picked this date because it is the birthday of John McCarthy, who had coined the term artificial intelligence!) The top three winners from each of the two categories (Best Performance Overall and Most Innovative) will be invited to a private event with Andrew Ng to share ideas about how to grow the data-centric movement, and will be highlighted in The Batch and other DeepLearning.AI and Landing AI channels. Good luck!

In this tutorial, we will cover the process of submitting your dataset and results for official evaluation on the Data-Centric AI Competition. Once your dataset has been evaluated officially, your scores will be added to the[**【leaderboard】**](https://landing-ai.github.io/data-centric-comp/). Before we begin, please create a CodaLab Worksheets account. Be sure to create a[**【CodaLab Worksheets account】**](https://worksheets-dev.codalab.org/), not to be confused with a standard CodaLab account.   

![leader board](https://github.com/Nov05/deeplearningai-data-centric-competition/blob/20217a8ba3b661d4796cd2f776f1e0fbc12c5950/images/leader%20borad.jpg)

### Rules
* Maximum 5 submissions per week (65 total over the course of the competition)  
* Submission must have less than 10,000 images combined in training and validation  
* Only one CodaLab account per team  

Submissions will be evaluated according to two categories:  
* Best Performance Overall  
* Most Innovative  

### Getting started

First, download the dataset by clicking into the bundle (click the uuid 0xcea1d7) and press the download button. The dataset contains ~3000 images of handwritten roman numerals 1-10. Your task is to optimize model performance by improving the dataset and making training and validation splits.

![dataset samples](https://github.com/Nov05/deeplearningai-data-centric-competition/blob/97c42ccfa63fc30f53b783a0df10eea9996d3daa/images/dataset_samples.png)

Your submission should have the following file structure, with your training and validation data split into different folders. You can rename sample_submission whatever you like, but the file structure and names inside must match the following:   
```
sample_submission/
    train/
        i/
        ii/
        iii/
        iv/
        ...
    val/
        i/
        ii/
        iii/
        iv/
        ...
```
You can try fixing incorrect labels, adding data for side case tuning, apply data augmentation techniques, or use any other method to improve the data. You may also find it helpful to take a look at the training script to get a better sense of the preprocessing and model (these are held fixed). The script will resize all images to (32, 32) and run them through a cut off ResNet50.   
* train.py   

If you choose to create your own data, you may find this script helpful for converting your images:  
* convert.py   

We also provide you with a label book that has five examples for each label. You can use it as a sample test set as you create your submission:  
* label_book   

### Submission  

Because we use automated evaluation to score your submission, please follow the steps in this tutorial exactly to ensure proper submission.

**Step 1: Upload improved training data**     
Go to your CodaLab profile and make a new worksheet.    
Name it whatever you like, and upload your submission folder by clicking the upload button on the tool bar. Make sure you upload the folder as a zip file.    

**Step 2: Copy standard bundles into your worksheet**    
Go into the web interface terminal (click the show terminal button on the right side of the toolbar) and enter the following commands:   
`cl add bundle data-centric-utils//train.py `   
`cl add bundle data-centric-utils//label_book`   

**Step 3: Make a run bundle to get predictions on the label book**    
Next, train the standard model on your submission data to make predictions on the label book. Use the following command in the web interface terminal:   
`cl run :train.py :<submission-file-name> :label_book 'python train.py <submission-file-name> label_book' -n run-train --request-docker-image jupyter/tensorflow-notebook`  

Be sure to change the <submission-file-name> field to the name of your folder. For example, for if you uploaded a zip file named sample_submission.zip, you would change the fields to sample_submission:    
`cl run :train.py :sample_submission :label_book 'python train.py sample_submission label_book' -n run-train --request-docker-image jupyter/tensorflow-notebook`   

You should see a run-bundle like the following appended to the contents of your worksheet. If the bundle fails, reach out on Discourse and we can help you out.   
* run-predict2   

**Step 4: Extract predictions file**   
Next, extract out the predictions file into a bundle of its own. Run the following command in your web interface terminal:   
`cl make run-train/predictions.json -n <your-submission-name>`    

Your <your-submission-name> should not contain spaces (avoid using special characters too). For example: cl make run-train/predictions.json -n sample-submission. You should see a bundle like the following appended to the contents of your worksheet:   
* sample-submission   

![codalab worksheets](https://github.com/Nov05/deeplearningai-data-centric-competition/blob/db48907a6420c4daf333d71f672658ba0d06c2c1/images/CodaLab%20Worksheets.jpg) 

**Step 5: Tag your submission and add description**    
You're almost there! Give your submission a description. You must have a description for proper submission! First, click into the <your-submission-name> bundle by clicking the small arrow next to the uuid and modify the Description field on the right sidebar.   

Please use the following convention for the bundle description to identify your submission on the leaderboard:    
`<your-submission-name> (Institution) optional_http_link`     

An example of a properly formatted description:   
* baseline (X university) https://arxiv.org/abs/something    

You're now finally going to add the mnist-roman flag to mark the bundle as ready for submission. Run the following command:   
`cl edit <your-submission-name> --tags mnist-roman`   
 
**Step 6: Fill out Google Form**   
Fill out this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScpcF8UNtYKGkYUf0OPeWJxQJCC7zEjIl7E-huio7bRAhgnUw/viewform) so we can contact you and you are done!! Congratulations! Please give us a couple days to evaluate your submission and add it to the leaderboard.   

### More Information   

If you have any questions or want to form teams with others, join us on Discourse. Use [this link](http://bit.ly/dlai-competition) to join if you are a new user, and [this link](https://discourse.deeplearning.ai/g) if you are an existing user.    

Check out Andrew Ng's talk on shifting from a Model-Centric approach to a Data-Centric approach to AI: [MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo).    

### Have Questions?

Join us on Discourse to ask us questions and connect with other competitors!   

For new Discourse users, please click [here](http://bit.ly/dlai-competition) to create an account. After you sign in, you’ll find a “Competitions” category on the home page;   

For existing users, please click [here](https://discourse.deeplearning.ai/g) to log in, find the **"Data-centric AI Competition Participants"** group and join.    


  




 

    




