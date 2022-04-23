#!/bin/zsh

for year in {2019..2019}
do
  for i in {01..12}
  do
    unzip "$year$i-citibike-tripdata.csv.zip"
  done
done
