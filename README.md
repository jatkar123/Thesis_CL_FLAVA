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

   (v) AvalancheCL_CUB200.ipynb --> To check why and how the frozen FLAVA learns better; I performed similar freezing and unfreezing variations as (iii) but with a much more complex S-CUB200 dataset which is more realistic and has around 5k training images of 200 species of birds in total. Unfrozen FLAVA models with Naive and ER CL strategy either learn very very slowly or stagnate after a few epochs (~2% accuracy). On the other hand Frozen FLAVA model with Naive shows decreasing loss and an accuracy of ~11% (no hyperparameter tuning). Again, frozen model with ER CL strategy shows a nice loss decreasing curve and an accuracy of ~36%. Therefore, I think the two layered model parameter update in un-frozen FLAVA (once in classification head and once again in FLAVA image model) confuses the entire CL architecture and it makes it difficult for it to learn. But, with only one layer of model paramater update in frozen model, CL works better here.
    * __Observation__ --> Frozen FLAVA works better than Un-frozen as mentioned in the CLIP frozen model paper ("CLIP MODEL IS AN EFFICIENT CONTINUAL LEARNER"). Why and reasoning experimented ahead.

   (vi) Frozen_part_flava_cub.ipynb --> To further ascertain as to why the Frozen model works better, the primary reason could be such huge amount of parameters in FLAVA can't change very quickly/rapidly (learn) with a small complex dataset like S-CUB200, So, to further support this, I partly froze FLAVA. On a high level overview, the Image model of FLAVA has 3 main children (patch embeddings, main encoder and pooler); so, I froze only the last layer/child of this image model and got better loss decreasing curve and much better accuracy than the completely frozen FLAVA model (~92% accuracy).
    * __Goal achieved__ --> Proved that more the parameters (in FLAVA), more difficult it gets in a CL setting for a pre-trained model to learn efficiently. Therefore, we can say that pre-trained models drive the weights to a wider minima to better generalize the model but with over 200 classes (with only 5k images i.e only 25 images per class). This generalized learning is very diifcult in such a case. But, in my opinion with more images per class like in mini image net dataset, this generalization can be better shown and a right combination between stability-plasticity can be found.
  
   (vii) Frozen_part_flava_cifar --> This is not an official notebook but basically inherits everything from (vi) and the only change is that the partly frozen FLAVA was experimented on CIFAR100 dataset in a similar fashion like 2 out of 3 child layers frozen to check if this also follows a similar pattern in loss curve learning like CUB200 dataset.
   
   *__Observation__ --> CIFAR100 also showed a similar learning curve where the partly frozen model worked better than the completely frozen one. The one major change was that the number of neurons in the MLP classifciation head had to be increased to embibe more learnings. To further prove that this learning is not just shown with loss curves and accuracies but we have calculated the CKA (Centered Kernel Alignment) measure which tells the similarity measure between different neural network layers and this too shows that there is similarity only in the learnings of the first layer but the rest two have learnt different representations in partly frozen scenario.

  
   



