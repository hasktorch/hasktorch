#!/usr/bin/env bash

mkdir -p data
wget https://raw.githubusercontent.com/nytimes/covid-19-data/master/excess-deaths/deaths.csv -O ./data/deaths.csv
wget https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv -O ./data/us-counties.csv