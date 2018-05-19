#!/bin/sh
python transform.py
wait
python sort_by_time.py 
wait
python create_cats.py &
python create_userDetail.py &
python createTimeDict.py &
#python ZHL.py &
#python cut_train.py &
wait
python commDig.py &
python clickDig.py &
python searchDig.py &
python snowDig.py &
python timeDig.py &

python top3Dig.py &

wait
python merge.py && python selct_train.py

wait
python LGB.py