# shell iter

## enter data dir and record all the csv file under this dir
cd /home/ZwZ/database/BRACA/leukocyte_ratio/
CSV=$(ls)

## enter script dir
cd /home/ZwZ/script/HER2_prediction/leukocyte/src/
## dir to where log files save 
LOG_PATH=/home/ZwZ/script/HER2_prediction/leukocyte/log/

## iter each file
for csv in ${CSV};
    do LOG="$LOG_PATH$csv.log"
    nohup python main_leuko_cluster.py --csv $csv > $LOG &
#echo $LOG
done