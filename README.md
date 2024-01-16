
Data Visualization and Analysis
The initial data set did not contain the class ‘good’ which represents the comment set that has all the given classifications as 0 (Non-toxic). As evident from the histograms below the majority of comments are actually “good” comments, a hidden class that would otherwise be undermined and overpredicted since it represents 0s in all other categories. This heavy imbalance in our dataset tended to push our model to predict Non-toxic for every single comment. We combated this by adding sample weights to each data point, giving a high weight to samples that had any toxic class as 1 and a low weight to those that had 0s for all classes. This didn’t have the impact we hoped for, as the model changed from predicting 000000 for all comments to 101010. The imbalance in the toxic classifications itself was also strong enough to completely remove them from being classified. This led us to also adding class weights, giving a higher weight to the class that had very little comments, `severe-toxic`, `threat`, and `identity_hate`. Surprisingly, this had no effect on the prediction besides altering the prediction from 101010 to 010101 for all comments. Any permutation of the weights would lead to a different permutation of classes being predicted for every single comment. This is when we thought of adding the extra class ‘good’ in our dataset, in hopes to increase complexity for our data. We believed this would help the model in realizing that comments classified as not ‘good’ would have some form of toxicity.
By understanding the context of the problem we also realized that it is much more important to filter out the toxic comments than to incorrectly filter out the good comments, which essentially means that in our predictions, the “false positive” (classifying an otherwise good comment as one of the toxic categories”) carries less weight than “false negative” (passing a toxic comment as good).

The technique we focused on for this project was the Transformer deep learning Architecture, specifically the BERT model. BERT is a bidirectional transformer which has been pre trained on a large corpus which includes the Toronto Book Corpus and Wikipedia. While the model is initially pre-trained, transformers typically require large amounts of fine-tuning for optimization towards specific tasks. The manner in which we train it with our own data significantly influences both training and testing performance. There are many possible ways to parameterize this model, such as adding layers on top of a generic BERT model, using a pretrained BERT model for multi-label classification, altering the tokenizer, loss function, optimizer, batch size, and introducing new class weights, batch weights, and labels. We were able to touch on most of these hyper parameters in our project. Tejasv was primarily responsible for exploring and displaying the data distribution, and choosing the evaluators that we used for our training, validation, and test sets. Rohan was primarily responsible for training the model by writing code for the dataset, data loader, model, and training. We had decided from the beginning that we would use accuracy, precision, recall, f1 score, and confusion matrices to evaluate our models. 
Our exploration started with understanding how to read and visualize the data from the Kaggle competition and exploring starter models posted on the code section on the Kaggle site. We were both familiar with transformers and Hugging Face, so when we saw numerous example notebooks using the BERT transformer, we decided it would be a good idea to try to use and explore this model. We planned to use the PyTorch Dataset and Dataloader classes when training, so we created our own PyTorch dataset which took in comments, labels, and weights, and returned the tokenized outputs accordingly. After exploring the BERT documentation, we learned of the BertTokenizer and BertForSequenceClassification, which is a BERT Model with a sequence classification head on top. Initially, we trained our model with base configurations to find a starting point. Because of the heavy imbalance in our dataset, unsurprisingly, the model predicted the negative class for all comments. We addressed this by introducing batch weights in our loss function, so we gave a large weight to data points that had any toxic label as 1, and a small weight to data points that had all toxic labels as 0. We configured weights accordingly so that predicting toxic labels as ‘000000’ for all comments would result in a 50% weighted accuracy. We adjusted our loss function by using PyTorch’s CrossEntropyLoss instead of the default loss produced by BERT, and set the `reduction` parameter to ‘none’. With reduction set to ‘none’, CrossEntropyLoss returns individual losses of each sample instead of computing the mean itself. Given a loss for each sample in the batch, we multiplied each loss by its weight and then took the mean to find the batch loss. Unexpectedly, overcoming the issues that followed this weighted model ended up being the most difficult part of our project. 
When running our batch weighted model numerous times with different batch sizes and learning rates, the test results were always the same: the model would predict the same classification for every comment. Our first weighted model predicted 101010 for every single comment. This was definitely unexpected, because while the batch weights did remove some imbalance from our dataset, we were unsure why it would predict the same classification for every comment. We understood that there was an imbalance in the different classes of toxicity as well, but that wasn’t the root cause of every comment being classified the same way. The labels that predicted true everytime had thousands of comments, and those that were predicted false had only hundreds. This is when we introduced class weights, so the classes that had only hundreds of comments were given a higher weight. We did this by setting the `weight` parameter in CrossEntropyLoss alongside the reduction one. However, altering the weight for the loss function led to no improvement, but instead with different classifications for every weight. For example, dramatically increasing the weights for the labels that were always predicted negatively led to the prediction changing from 101010 to 010101 for all data points. Reducing or rearranging these weights would only lead to different permutations being predicted every time. We eventually came to the conclusion that our data wasn't complex enough, so we decided to add a new label to our model. We called this new label ‘good’, and it was 0 for comments where all classifications were 0 and 1 if any of the classifications were 1. Our intuition with this new label was to adjust the class weights so that the loss would be better for correctly predicting the original 6 labels, but it would still take into account the labels that didn’t have any toxicity. We were able to easily add this label to our data using pandas, and then ran the model with 7 labels and class weights. Also, one thing we noticed while training was that our model seemed to converge rather quickly, so we reduced our train time to only 1 epoch instead of 5.
Finally, the model produced test results where it differently classified individual comments. On top of that, when we ran it without class weights, it performed even better. We were able to increase the batch size to 16, the largest batch size that could fit on Kaggle’s GPU, in hopes to get at least 1 or 2 toxic comments in every batch. This was the 4th model we tried, and although only 24% of all comments were correctly predicted for all 7 labels, 82% of all individual labels were predicted correctly and our model had a 0.68 weighted f1 score and 0.97 weighted recall. Our model was overpredicting for toxic classes which we could see because of a precision of 0.55 and with our confusion matrices. Our validation error and losses throughout this process were always very similar to the final training error and loss. 
After adding the new label, we thought that the next step to improve our model would be to further the model complexity. A few ideas we thought of were to add a one hot encoding for all the combinations of classifications, in hopes to have an effect similar to our one new ‘good’ class, or to add more complexity to the head of the BERT Model by making our own classification head instead of the one pre configured by the BertForSequenceClassification model. We chose to follow through with the latter because we thought that it had more potential for improvement. Our best model so far was the one where we added a linear, tanh, and another linear layer on top of the last hidden-state layer to predict the logits. Prediction for this model was done the same as always, by applying a sigmoid activation function on those logits and then rounding. This model had the best results, with a weighted accuracy of 67%, f1 score of 0.65, and a weighted f1 score of 0.8. It was also able to predict almost 97% of all individual labels correctly. This model also tended to overestimate toxic comments, so while the weighted precision was 0.8167, the normal precision was only 0.5578. However, having a low precision for this specific problem is better than having a low recall, which was around 0.78. Going forward, we plan to adjust class weights to reduce overestimation for toxic labels in general and also to improve classification for the smaller classes, such as `threat` and `identity_hate`. This is important because those are the most important classes to identify accurately due to their toxicity. Adding further complexity at the head of the model should also improve results as well. 
Evaluation
Model Num.
Model
Batch size
Num of labels
Loss Function
Test Accuracy
- Accuracy: fraction of data with all predicted labels correct
- Total Accuracy: fraction of labels correct
Test Precision
Test Recall
Test F1 Score
Train Error
Train Loss
Validation Error
Validation Loss
1
BertForSequenceClassification
1
6 (original)
BERT Loss (Cross Entropy Loss)
Weighted Accuracy: 0.5000
Weighted Precision: 0.0
Weighted Recall: 0.0
Weighted F1 score: 0.0
Train Error: 0.15
Validation Error: 0.19
Train Loss: 0.023
Validation Loss: 0.024
2
BertForSequenceClassification
8
6
Cross Entropy Loss with batch weights
Weighted Accuracy: 0.0291


Weighted Precision: 0.0691
Weighted Recall: 0.9085
Weighted F1 Score: 0.1284
Train Error: 0.47
Validation Error: 0.48
Train Loss: 2.6
Validation Loss: 2.7
3
BertForSequenceClassification
16
6
Cross Entropy Loss with batch and class weights
Weighted Accuracy: 0.0291


Weighted Precision: 0.0691
Weighted Recall: 0.9085
Weighted F1 Score: 0.1284
Train Error: 0.49
Validation Error: 0.51
Train Loss: 0.48
Validation Loss: 0.5
4
BertForSequenceClassification
16
7 
Cross Entropy Loss
Unweighted Accuracy: 0.2405
Weighted Accuracy: 0.2492
Weighted Total Accuracy: 0.8192
Weighted Precision: 0.5175
Weighted Recall: 0.9723
Weighted F1 Score: 0.6755
Train Error: 0.15
Validation Error: 0.13
Train Loss: 0.45
Validation Loss: 0.36
5
Bert with Classification Head (Linear, Tanh, Linear)
16
7
BCE With Logits Loss
Unweighted Accuracy: 0.8668
Weighted Accuracy: 0.6686
Weighted Total Accuracy: 0.9248
Weighted Precision: 0.8167
Weighted Recall: 0.7884
Weighted F1 Score: 0.8023
Trian Error: 0.02
Validation Error: 0.02
Train Loss: 0.04 
Validation Loss: 0.05

Confusion Matrices, Loss, and Error Rate
Model 2
Model 4
Model 5
---Confusion Matrices---
[[true_negative, false_positive]
 [false_negative true_positive]]
Confusion Matrix for toxic: 
[[    0 57888]
 [    0  6090]]
Confusion Matrix for severe_toxic: 
[[63611     0]
 [  367     0]]
Confusion Matrix for obscene: 
[[    0 60287]
 [    0  3691]]
Confusion Matrix for threat: 
[[63767     0]
 [  211     0]]
Confusion Matrix for insult: 
[[    0 60551]
 [    0  3427]]
Confusion Matrix for identity_hate: 
[[63266     0]
 [  712     0]]

    ---Confusion Matrices---
[[true_negative, false_positive]
 [false_negative true_positive]]
Confusion Matrix for toxic: 
[[13761 44127]
 [    0  6090]]
Confusion Matrix for severe_toxic: 
[[62503  1108]
 [   61   306]]
Confusion Matrix for obscene: 
[[52316  7971]
 [   82  3609]]
Confusion Matrix for threat: 
[[63692    75]
 [  151    60]]
Confusion Matrix for insult: 
[[49089 11462]
 [   28  3399]]
Confusion Matrix for identity_hate: 
[[61841  1425]
 [   79   633]]


       ---Confusion Matrices---
[[true_negative, false_positive]
 [false_negative true_positive]]
Confusion Matrix for toxic: 
[[52854  5034]
 [  514  5576]]
Confusion Matrix for severe_toxic: 
[[63525    86]
 [  288    79]]
Confusion Matrix for obscene: 
[[58221  2066]
 [  632  3059]]
Confusion Matrix for threat: 
[[63767     0]
 [  210     1]]
Confusion Matrix for insult: 
[[58685  1866]
 [  771  2656]]
Confusion Matrix for identity_hate: 
[[63257     9]
 [  653    59]]



Resources
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification 
https://discuss.huggingface.co/t/class-weights-for-bertforsequenceclassification/1674/2 
https://discuss.pytorch.org/t/weighted-cross-entropy-for-each-sample-in-batch/101358/9
https://www.kaggle.com/code/samarthagaliprasad/bert-for-toxicity 
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=B9_DjWmfWx1q  
https://github.com/AdeelH/pytorch-multi-class-focal-loss 

