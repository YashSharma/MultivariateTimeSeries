# Encoding Multivariate Time Series as Images for Classification using Convolutional Neural Network

Exercise testing has been available for more than a half-century and is a remarkably versatile tool for diagnostic and prognostic information of patients for a range of diseases, especially cardiovascular and pulmonary. With rapid advancements in technology, wearables, and learning algorithm in the last decade, its scope has evolved.
Specifically, Cardiopulmonary exercise testing (CPX) is one of the most commonly used laboratory tests for objective evaluation of exercise capacity and performance levels in patients. CPX provides a non-invasive, integrative assessment of the pulmonary, cardiovascular, and skeletal muscle systems involving the measurement of gas exchanges. However, its assessment is challenging, requiring the individual to process multiple time series data points, leading to simplification to peak values and slopes. But this simplification can discard the valuable trend information present in these time series. In this work, we encode the time series as images using the Gramian Angular Field and Markov Transition Field and use it with a convolutional neural network and attention pooling approach for the classification of heart failure and metabolic syndrome patients. Using GradCAMs, we highlight the discriminative features identified by the model. 

# Reference

Accepted at NeurIPS 2021-MLPH Workshop and EMBC 2022 conference: https://arxiv.org/abs/2204.12432
