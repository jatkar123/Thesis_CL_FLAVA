# Thesis_CL_FLAVA
This repository is dedicated towards keeping the code infrastructure for my Master Thesis--> "How Continual Learning affects FLAVA"
Insert more about the topics here...

## Folder Structure and Notebook Description:  
1) Image Classification FLAVA:<br>
   
   (i) Flava_ImageClassification_MNIST_with_comments.ipynb--> This notebook explores inferencing of FLAVA Image encoder model (i.e ViT B-16 model) on MNIST and CIFAR-10 dataset with a logistic regression classifier head (as mentioned in the FLAVA paper). On both these datasets the FLAVA model gives great results (~95%+ accuracy) in zero-shot inference but one thing which needs to be taken into consideration is that FLAVA  model has already been trained on these models, so this doesn't completely justify its efficacy.
   * __Goal achieved__--> Learnt how the FLAVA model API's work and how exactly is it structured and how I can modify it accoring to my thesis requirements. And how to fit and run inference on complete FLAVA using batching on one 32GB GPU.

   (ii) Flava_IM_on_3d_print_dataset.ipynb --> This notebook is an extension of  (i) which tests on how zero-shot inference works on FLAVA for a dataset which it has never before seen. I used the 3d printed image dataset which is a medium sized (60k training images) fairly complex dataset. With the same classification head and outline as above, the model got a 94.44% train set accuracy and 80% test set accuracy.
   * __Goal achieved__-->This notebook showcases that inference on FLAVA on a new dataset also works well with FLAVA model. Now, we can move onto experimenting with continual learning strategy altogether on FLAVA.

