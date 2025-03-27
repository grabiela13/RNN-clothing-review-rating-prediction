# Rating Prediction for Women's E-commerce Clothing Reviews with RNN

## Project Overview
This project predicts product ratings based on review text using Recurrent Neural Networks (RNNs). The goal is to analyze the sentiment of reviews from a women's e-commerce clothing dataset and predict ratings ranging from 1 to 5 stars. The project uses **TF-IDF** and **Word Embeddings** for feature engineering and compares the performance of two RNN models.

## Dataset
The dataset contains the following attributes: Clothing ID, Age, Title, Review Text, Rating, and others. For this project, we focus on the **Title**, **Review Text**, and **Rating** columns to predict ratings using the review text.

## Methodology
1. **Data Preprocessing**: 
   - Combine the review title and text.
   - Clean the text by converting to lowercase, removing punctuation, and eliminating stop words (except "no" and "not").
   - Normalize the ratings to a 0-1 scale.
   
2. **Feature Engineering**: 
   - **TF-IDF**: Converts the text into numerical features based on word importance.
   - **Word Embeddings**: Represent words in continuous vector space, capturing semantic relationships.

3. **Model Development**: 
   - Two RNN models were built: one using TF-IDF features and the other using word embeddings.
   - The models used a **Bidirectional GRU layer** and **Dense layer** for rating prediction.

4. **Evaluation**: 
   - The models were evaluated using **Mean Absolute Error (MAE)** for rating prediction.
   - A **binary classification** experiment was conducted to predict whether a review was 5-stars or not, achieving an accuracy of 81.23%.

## Results
- **TF-IDF Model MAE**: 0.1311 (after 8 epochs)
- **Word Embedding Model MAE**: 0.1326 (after 9 epochs)
- **Binary Classification Accuracy**: 81.23% (TF-IDF), 80.85% (Word Embeddings)

## Conclusion
This project demonstrates the power of RNNs in predicting product ratings from review text. TF-IDF slightly outperformed word embeddings in predicting ratings, while both methods performed well in binary classification tasks. Further improvements could include exploring other model architectures or additional preprocessing steps.


## References
1. [Women's E-Commerce Clothing Reviews Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data)
2. [Disneyland Review Rating Prediction](https://www.kaggle.com/code/gcdatkin/disneyland-review-rating-prediction)
3. [Matplotlib](https://matplotlib.org/)
