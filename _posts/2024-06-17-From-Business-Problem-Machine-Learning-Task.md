---
title: "Transforming Business Problems into Machine Learning Solutions"
date: 2024-06-17
summary: This post explores the process of identifying whether a business problem can be effectively solved using machine learning. It delves into key considerations, such as impact and cost, and provides guidelines on framing ML problems, defining goals, and selecting appropriate models for high-impact, low-cost solutions.
categories:
- Frame an ML Problem
- High-Impact and Low-Cost
---
## Is ML is the Right Approach?

An example which i read from the book 'Machine Learning Engineering in Action' really illustrate this well. When a data scientist, initially thinking they need to implement an advanced machine learning model to enhance email marketing effectiveness, discovers through careful questioning that the actual requirement is much simpler. Instead of building a content recommendation system or designing NLP-driven subject lines, the real solution needed is a straightforward analytics query. By identifying the users' time zones based on IP geolocation, the company can optimize email send times to increase open rates. This really requires engineers to understand the gap between business problems and machine learning solutions.

**A Good ML Project:**

High-Impact:

1. Cheap prediction will have a high business impact. Cheap prediction means that prediction will be everywhere; even in problems where it was too expensive before.
2. If a part of a sytem that is complex and manually defined, then that's potentially a good candidate to be automated with ML.

Low-Cost:

1. Data is largely available
2. Bad predictions are not too harmful
3. Relatively low maintenance

**Types of problems are still hard to solve with ML**

* Output is complex, output is high-dimenisonal and ambiguous. Such as 3D reconstruction, video prediction, open-ended recommender
* Reliability is required. High precision and robustness are required. Such as failing safely out-of-distribution, high-precision pose estimation
* Generalizaiton is required. Out of distribution data, reasoning, planing and causality. Such as edge cases and control for self-driving, small data

**Think about it from those aspects:**

* State the goal for the product you are developing or refactoring.
* Determine whether the goal is best solved using, predictive ML, generative AI, or a non-ML solution.
* Verify you have the data required to train a model if you're using a predictive ML approach.

## Frame an ML Problem

#### State the Goal

First state the goal in non-ML terms. The goal is to answer the question,  "What am i trying to accomplish here?"

| Apps                 |     | Goal                                                           |
| -------------------- | --- | -------------------------------------------------------------- |
| Healthcare app       |     | Diagnose medical conditions based on symptoms.                 |
| Customer service app |     | Automate responses to common customer queries.                 |
| Fitness app          |     | Track and analyze workout progress over time.                  |
| Food delivery app    |     | Predict delivery times based on traffic conditions.            |
| Real estate app      |     | Match properties with potential buyers based on preferences.   |
| Education app        |     | Assess student performance and suggest improvement strategies. |
| Recipe app           |     | Suggest recipes based on available ingredients.                |
| Mail app             |     | Detect spam.                                                   |

Clear Use Case for ML

|               | input                                                 | output                                                                                                            | training techniques                                                                                                               |
| ------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Predictive ML | Text<br />Image<br />Audio<br />Video <br />Numerical | Make a prediction. Predict the house price based on certain features;<br />classify spam or non spam for an email | Typically uses lots of data to train a supervised, unsupervised,<br />or reinforcement learning model to perform a specific task. |
| Generative ML | Text<br />Image<br />Audio<br />Video<br />Numerical  | Generate output based on prompt                                                                                   | Typically uses lots of unlabeled data to train a LLM or fine tune<br />pre-trained models                                         |

#### Define the ideal outcome and the model's goal

| Apps                 | Goal                                                           | Models Goal                                                      |
| -------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| Healthcare app       | Diagnose medical conditions based on symptoms.                 | Predict what types of medical condition based on symptoms        |
| Customer service app | Automate responses to common customer queries.                 | Generate response to customer querires                           |
| Fitness app          | Track and analyze workout progress over time.                  | Predict whether a user will continue using the app               |
| Food delivery app    | Predict delivery times based on traffic conditions.            | Predict how long it will be deliverd based on traffic conditions |
| Real estate app      | Match properties with potential buyers based on preferences.   | Predict whether a buyer will purchase certain properties         |
| Education app        | Assess student performance and suggest improvement strategies. | Predict whether a student will chhose a strategy                 |
| Recipe app           | Suggest recipes based on available ingredients.                | Predict which recipes a user will choose based on ingredients    |
| Mail app             | Detect spam.                                                   | Predict whether a mail is spam                                   |

#### Identify the model's output

Predictive ML

| Task Type      | Specific Task             | Model Type                                                                                              | Description                                                                                           |
| -------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Classification | Binary Classification     | Logistic Regression, Bayes classifier, SVM, Random Forest,<br />Gradient Boosting, Neural Networks      | Predict two possible outcomes (e.g., spam detection)                                                  |
|                | Multiclass Classification | Multinomial/Multi-class Logistic Classification, Multi-class Decision Tree,<br />Gaussian Mixture Model | Predict more than two possible outcomes (e.g., handwritten digit recognition).                        |
|                | Text Classification       | Logistic Regression, SVM, RNNs, CNNs, Transformers                                                      | Classify text into categories (e.g., sentiment analysis).                                             |
| Regression     | Simple Regression         | Linear Regression, Ridge/Lasso Regression                                                               | Predict continuous values (e.g., house prices).                                                       |
|                | Complex Regression        | Decision Trees, Random Forest, Gradient Boosting, Neural Networks                                       | Predict continuous values with more complex relationships.                                            |
| Clustering     | K-Means Clustering        | K-Means, MiniBatch K-Means                                                                              | Group similar data points (e.g., customer segmentation).                                              |
|                | Hierarchical Clustering   | Agglomerative Clustering, Divisive Clustering                                                           | Create a hierarchy of clusters (e.g., document clustering).                                           |
|                | Density-Based Clustering  | DBSCAN, OPTICS                                                                                          | Find clusters of arbitrary shape and handle noise (e.g., geographic data clustering).                 |
|                | Model-Based Clustering    | Gaussian Mixture Models (GMM)                                                                           | Assume data is generated from a mixture of several Gaussian distributions (e.g., speech recognition). |

Generative AI

| Task Type                | Specific Task                                    | Model Type                | Description                                                                                      |
| ------------------------ | ------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------ |
| Text Generation          | Creating coherent and contextually relevant text | Transformers (GPT, Llama) | Generate text for applications like chatbots, story generation, and text completion.             |
| Video Generation         | Creating new video sequences                     | GANs, RNNs, Transformers  | Generate video content for applications like deepfake creation, video prediction, and animation. |
| Data Augmentation        | Enhancing datasets with generated data           | GANs, VAEs                | Generate synthetic data to augment training datasets for improving model performance.            |
| Text-to-Image Generation | Creating images based on text descriptions       | DALL-E, AttnGAN           | Generate images from textual descriptions for applications like illustration and design.         |

##### Classification vs Regression

A [**classification model**](https://developers.google.com/machine-learning/glossary#classification-model) predicts what category the input data belongs to, for example, whether an input should be classified as A, B, or C. Based on the model's prediction, your app might make a decision. A [**regression model**](https://developers.google.com/machine-learning/glossary#regression-model) predicts a numerical value. Although sometime you can frame your problem as classification or regression, one thing is worth noting is that Regression models are unaware of **product-defined thresholds**. Therefore, if your app's behavior changes significantly because of small differences in a regression model's predictions, you should consider implementing a classification model instead.

Predict the decision. When possible, predict the decision your app will take. For example, a classification model would predict the decision if the categories it classified videos into were "no cache," "cheap cache," and "expensive cache." Hiding your app's behavior from the model can cause your app to produce the wrong behavior.

Understand the problem's constraints: If your app takes different actions based on different thresholds, determine if those thresholds are fixed or dynamic.

* Dynamic thresholds: If thresholds are dynamic, use a regression model and set the thresholds limits in your app's code. This lets you easily update the thresholds while still having the model make reasonable predictions.
* Fixed thresholds: If thresholds are fixed, use a classification model and label your datasets based on the threshold limits.

##### Proxy labels

In the above education app example, which recommends a study strategy for a student, there isn't a label called "useful_to_student." Therefore, we need to find a proxy label that substitutes for usefulness, such as "liked." No proxy label can be a perfect substitute for your ideal outcome. All will have potential problems. Pick the one that has the least problems for your use case.

##### Generation models

1. Distillation: To create a smaller version of a larger model, you generate a synthetic labeled dataset from the larger model that you use to train the smaller model. Generative models are typically gigantic and consume substantial resources (like memory and electricity). Distillation allows the smaller, less resource-intensive model to approximate the performance of the larger model.
2. Fine-tuning: To improve the performance of a model on a specific task, you need to further train the model on a dataset that contains examples of the type of output you want to produce.
3. Prompt engineering: To get the model to perform a specific task or produce output in a specific format, you tell the model the task you want it to do or explain how you want the output formatted

#### Define the Success Metrics

Success metrics are crucial for determining the effectiveness of your machine learning implementation. These metrics are different from traditional evaluation metrics like accuracy, precision, recall, or AUC. They focus on the practical impact of the ML model on user behavior and overall business goals.

Technically its more like measuring the success of the ML product rather than measuring the ML model.

* Define quantative metric such as number of users a feature attracts or the click-thorugh rate(CTR)
* User Satisfaction survey
