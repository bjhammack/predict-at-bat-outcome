# Predict At Bat Outcome

Baseball has always been a cutting edge sport in regards to analytics. More and more predictive analytics is being used to improve player performance and team decision making. In the spirit of that, this project aims to give better insight on the quality of a hit by training a neural network to predict the outcome of an at bat, based on the pitch velocity, exit velocity, and launch angle.


# Data

The dataset is every player who was active in 2020's at bats from 2015 - 2020. This totals <> rows of data, almost all of which is very high quality. The data was acquired via another program I've written, that scrapes baseball statistics off of various sites (mlb.com, fangraphs.com, baseballsavant.mlb.com, etc.) and saves them for use later. These particular statistics (pitch velo, exit velo, and launch angle) were scraped from baseballsavant.mlb.com.