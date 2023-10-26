# Chirper Sentiment Analysis

## Project Overview

This project is a part of the final test for a Naive Bayes Classifier to earn a certificate. It involves building a sentiment analysis model to categorize Chirper posts(platform) as positive (Chirps) or negative (Caws).

## Project Context

### The Chirper Platform

Chirper is a social platform where users communicate by uploading posts on various topics.
They can express their excitenment over a great song, an upcoming movie,etc. Posts can express excitement,However sometimes users feel displeased, angry or disappointed too.

## Dataset Overview

Data Size: The dataset consists of 10,000 rows and 6 columns.
The dataset(chiper.csv file) is stored in a pandas DataFrame and contains the following columns:

1. **Target:** This column signifies the sentiment of each post.
   - **0:** Represents 'Caws' (negative posts).
   - **1:** Represents 'Chirps' (positive posts).

2. **IDS:** A unique identifier for each post.

3. **Date:** The timestamp indicating when the post was made.

4. **Flag:** An indicator or flag associated with the post, if applicable.

5. **User:** The username or account linked to the post.

6. **Text:** The textual content of the posts.
   
#Only 2 of these columns are necesary to perform the classification: 'Target' and 'Text'.

 **Target:** This column signifies the sentiment of each post.
   - **0:** Represents 'Caws' (negative posts).
   - **1:** Represents 'Chirps' (positive posts).

## Model Selection

Choosing the Complement Naive Bayes algorithm due to the dataset's small size and severe class imbalance.

## Justifying Model Performance

1. The model could use a larger training dataset compared to the current one.
2. Some data points in the training or test data might be labeled incorrectly.

**Explanation**:
- The Complement Naive Bayes algorithm is designed for imbalanced datasets, making a larger dataset beneficial.
- The dataset is quite small and severely imbalanced. It is not trained optimally on the caw class.Therefore a larger dataset would definitely improve the performance.
- We cannot exclude the fact that there might been a mistake during the data collection process.
-The random_seed should be of very little importance to the performance of the model
- The possibility of incorrectly labeled data points cannot be ruled out in the data collection process.
- The random seed number is of little importance for model performance, whereas stratification is crucial for balanced class distribution during train-test splitting.
- A more important parameter for the sppliting would be stratify which makes sure that the classes are equally distributed among the training and testing sets. That is, we dont want to train mainly on chirps or caws respectively because of bad suffling.

## Model Comparison

At first, a Multinomial Naive Bayes model was considered. However, it was later switched to the Complement Naive Bayes model, which showed improved precision and recall for the 'Caws' class.
Identifying class 1 as positive the number of true positivies has decrease compared to the Multinomial.

## Conclusion

According to the classification report, there are also no false negative samples (here, 0 is referred to as the negative class)
From the classification report, we read off that the accuracy is well above 90%.
This result indeed looks very impressive, until we take a look at the confusion matrix,which reveals that no caws whatsoever have been identified.
The botton left cell storing the false negatives(predicted 0s but are 1s in reality.)
The fact that there are no true negatives and no false negatives leads to the following problem with the precision of the caw class.

Precision = True Positivies/ True Positivies + False Positivies =  0/0 =0 = Undefined 

In the formula we indentify 0 as the positive class in the formula/
The additional parameter zero_division corrects this. if the parameter is not there, the classification_report() method would throw an error but would still assign a value of 0 for the precision.
Using 0 might be appropriate in some cases, but in others, you might want to use a different value depending on the specifics of your analysis.

We first did with Multinomial Model, now we changed to complement to see diference.
We can see that the accuracy has actually decreased slighly, however the precision and recall 
metrics for the caws have indeed improved. It is also true that the number of true positivies,
shown on the bottom right corner of the confusion matrix, has decreased slighly.
From 2850 to 2810

Finally the F1 score is evaluated as follows:
#F1 = (2/ 1/precision + 1 /recall)