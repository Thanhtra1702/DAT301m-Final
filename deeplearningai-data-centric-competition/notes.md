

**Different accuracies (Local training vs CodaLab)**    
https://discourse.deeplearning.ai/t/different-accuracies-local-training-vs-codalab/15892/3  

local:   
val: 67.4%    
test: 50.0%   

codalab   
val: 67.5%   
test: 51.9%    


**How does data augmentation come in?**      
https://discourse.deeplearning.ai/t/how-does-data-augmentation-comes-in/15370    

jtrcos: It appears you are allowed to expand the size of the data set up to the limit of 10,000 train/validation images. So one approach is to add augmented data to the original data set which has 2,880 train/validation images.


**The validation dataset**   
https://discourse.deeplearning.ai/t/the-validation-dataset/15631    

ragar66: The model which is fixed is trained over 100 epochs. The validation set will be used to save the epoch which has the best model performance on it.       