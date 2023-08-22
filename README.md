# Thesis_CL_FLAVA
This repository is dedicated towards keeping the code infrastructure for my Master Thesis--> "How Continual Learning affects FLAVA"
Insert more about the topics here...

## Folder Structure and Notebook Description:  
1) Image_Classification_FLAVA:<br>
   
   (i) Flava_ImageClassification_MNIST_with_comments.ipynb--> This notebook explores inferencing of FLAVA Image encoder model (i.e ViT B-16 model) on MNIST and CIFAR-10 dataset with a logistic regression classifier head (as mentioned in the FLAVA paper). On both these datasets the FLAVA model gives great results (~95%+ accuracy) in zero-shot inference but one thing which needs to be taken into consideration is that FLAVA  model has already been trained on these models, so this doesn't completely justify its efficacy.
   * __Goal achieved__--> Learnt how the FLAVA model API's work and how exactly is it structured and how I can modify it accoring to my thesis requirements. And how to fit and run inference on complete FLAVA using batching on one 32GB GPU.

   (ii) Flava_IM_on_3d_print_dataset.ipynb --> This notebook is an extension of  (i) which tests on how zero-shot inference works on FLAVA for a dataset which it has never before seen. I used the 3d printed image dataset which is a medium sized (60k training images) fairly complex dataset. With the same classification head and outline as above, the model got a 94.44% train set accuracy and 80% test set accuracy.
   * __Observation__-->This notebook showcases that inference on FLAVA on a new dataset also works well with FLAVA model. Now, we can move onto experimenting with continual learning strategy altogether on FLAVA.
  
2) Continual_Learning_Flava_Avalanche: <br>

   (i) AvalancheCL_EWC_SMNIST_DIL.ipynb --> I explored how Avalanche CL library can be integrated with FLAVA model since there is no in-built type and timm library doesn't support 'FLAVA' model. A new class for FLAVA was created which returns only the image model (ViT) of FLAVA specifically only its final layer of learned embeddings. Also, experimented how out of the box ViT performs using timm library with Avalanche CL.
   * __Goal achieved__--> Sucessfully created a singular pipeline which works with Avalanche CL library with FLAVA. Also observed that the FLAVA model was not able to learn as well as out of the box timm based ViT model on the same S-MNIST dataset via EWC CL strategy using same hyperparameters.
  
   (ii) AvalancheCL_EWC_variations.ipynb --> In this notebook I tried to explore what exactly would help FLAVA model integrate in a CL environment. So, I tried using the logits which is basically passing the embeddings through a fully connected layer. This did not work well in DIL setting but the model was learning in a TIL setting. Further on, I experiemnted with custom made training functions to mimic CL and that too wasn't performing well.
   * __Observation__ --> Need to build a classifcation head on top of FLAVA custom class and then use those logits with Avalanche CL.

   (iii) AvalancheCL_EWC&Naive_frozen_unfrozen_variations.ipynb --> This notebook explores two things, firstly does taking all the dimensions of the embeddings (unlike inference) work better. (Yes it does with Naive CL). Secondly, does freezing the FLAVA model have an affect? And experiments clearly showed that frozen worked well with EWC and Naive CL strategy oth but un-frozen FLAVA only worked well with Naive CL strategy. All experiments were done on S-MNIST dataset in DIL incremental setting.
   * __Observation__ --> Frozen model works well (need to explore why?) and using the complete embedding with a simple MLP does wonders. But, EWC CL strategy doesn't seem to work well (and same is shown in literature too-> support the clause).

   (iv) AvalancheCL_Replay.ipynb --> Further experiments to support that freezing the FLAVA model works better even with ER CL strategy but maybe the reason is the simple S-MNISt dataset? Thus, that is further explored.

   (v) AvalancheCL_CUB200.ipynb --> To check why and how the frozen FLAVA learns better; I performed similar freexing and unfreezing variations as (iii) but with a much more complex S-CUB200 dataset which is more realistic and has around 5k training images of 200 species of birds in total. Unfrozen FLAVA models with Naive and ER CL strategy either learn very very slowly or stagnate after a few epochs (~2% accuracy). On the other hand Frozen FLAVA model with Naive shows decreasing loss and an accuracy of ~11% (no hyperparameter tuning). Again, frozen model with ER CL strategy shows a nice loss decreasing curve and an accuracy of ~36%. Therefore, I think the two layered model parameter update in un-frozen FLAVA (once in classification head and once again in FLAVA image model) confuses the entire CL architecture and it makes it difficult for it to learn. But, with only one layer of model parmater update in frozen model, CL works better here.

  
   



