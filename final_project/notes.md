# Final Project

Welcome to my final project for the CSCI 378 Deep Learning class that I took at
Reed College in Spring 2025.
The overarching goal of the associated neural network was to model the Earth's
temperature at a given latitude given past data.
I'm using a data set from Berkeley which can be found under the `data/`
directory.
This data set includes temperature information about certain days in the year
and is rather large.
The goal of my neural network is to be able to extrapolate what future
temperatures might look like based on trends we see in the existing data.

## Data Set

- The data set is called the Berkeley Earth daily TAVG full data set.
  You can find it for yourself as a `.txt` file on the Berkeley Earth website.
- This data set contains five key fields:
  - **Date**:
    This is an identifier for when the data point was taken.
    Don't ask me how they were generated because I don't know either.
  - **Year**:
    This is the year the data point was collected in.
  - **Month**:
    This is the month within the year that the data point was collected in.
  - **Day**:
    This is the day of the month in which the data point was gathered.
  - **Day of Year**:
    The day of the year that the data point was gathered.
  - **Anomaly**:
    This is the big one; this describes the temperature deviation of that sample
    from the expected temperature of that day.
- These data points should give us everything we need to estimate future
  anomalies.

## Goals

- The primary goal of this project is to train a physics-informed neural network
  which, given a year and a day within that year, can predict the temperature
  anomaly of that specific day.
- Because this data set only goes up to 2022, we can use more recent data
  (2022-2025) as a way of validating and testing our network.
- A potentially more lucrative goal is to train a neural network which can
  predict temperatures at specific latitudes given a specific day of the year
  although the data set that I currently have does not contain the information
  necessary to do something like that.
- Maybe I could do wildfire data instead and use that to predict where future
  wildfires are going to occur.
