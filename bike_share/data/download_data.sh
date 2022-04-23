#!/bin/zsh

for year in {2017..2017}
do
  for i in {01..12}
  do
    /opt/homebrew/bin/wget "https://s3.amazonaws.com/tripdata/$year$i-citibike-tripdata.csv.zip"
  done
done
