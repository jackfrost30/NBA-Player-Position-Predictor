# NBA Player Position Predictor

This project uses a Support Vector Machine (SVM) model to predict the position of NBA players based on their statistical performance. The aim is to classify players into their respective positions (e.g., guard, forward, center) using various player stats.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)

## Introduction

Predicting the position of NBA players can provide insights into player roles and team dynamics. This project leverages machine learning techniques to classify players based on their stats, such as points, rebounds, assists, and more.

## Installation

To get started, clone the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/jackfrost30/nba-player-position-predictor.git
   cd nba-player-position-predictor

## Usage

1. Ensure you have the dataset of NBA player stats ready. You can use the provided nba2021.csv or any other dataset in a similar format.
2. Run the model training script:
   ```bash
   python predict.py

## Features

1. Uses SVM for classification of NBA player positions.
2. Includes data preprocessing and feature engineering.
3. Provides a simple interface for training and predicting.

## Results

Achieved accuracy: 86

Observations that led me to better accuracy,

1. I got rid of players who had Minutes Played Per Game(MP) less than 15 minutes. This helped me remove players who were outliers, basically players who played very less minutes(bench players) compared to other players(starters).
2. I dropped irrelevant features from the dataset such as "Player name", "Age" and "Team". I have also added some new derived features from existing features.
3. AST/PTS: I added a new ratio of Assists to Points, this improves the chances of predicting Point Guards(PGs) and Shooting Guards(SGs). Normally, Point Guards(PGs) have more Assists compared to Points as their primary objective is to be a team player. On the other hand, Shooting Guards(SGs) tend to have more Points compared to Assists as their main goal is to shoot accurately from 3-point range and mid-range. Because of this, PGs will have a high AST/PTS ratio and SGs will have a low AST/PTS ratio.
4. TRB/AST: This improves the chances of predicting PG/SG to C/SF/PF. This predicts if a player is Backcourt or Frontcourt. Backcourts usually refer to the guards and Frontcourt refers to the forwards and center. Normally, The Forwards(PF, SF) and Center will have very high Total Rebounds(TRB) in a game and very few Assists. On the contrary, The Guards(SG, PG) will have very few Rebounds and way more Assists. So, The Forwards will have a high TRB/AST ratio and the Guards will have a low TRB/AST ratio.
5. STL/BLK: This is similar to TRB/AST ratio where Guards usually have a high STL because they are quick. Forwards and especially the Center will have a very high BLK because they are very tall and guard the foul line(area under the basket). So, Guards will have a high STL/BLK ratio and Forwards/C will have a low STL/BLK ratio.
6. I have mapped the class labels into numerical values. PG: 1, SG: 2, SF: 3, PF: 4, C: 5. This makes it easier for the model to evaluate the target variables and to evaluate the performance of the model.
7. For Feature Selection, I have used Mutual Information Classification(MI). Each feature has an MI score and I have only picked the features which has an MI score greater than 0.1. This eliminates all the features deemed to be irrelevant.
8. I have split the dataset into 75% training data and the remaining testing data. I have also used StandardScaler() to standardize the features because some features are in a percentage. This is also done to give equal weight to all the features.
9. I have used GridSearch to calculate the best hyperparameters for my model(Linear SVC).
   






