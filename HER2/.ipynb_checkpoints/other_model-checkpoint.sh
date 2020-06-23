
MODEL=(LDA RF)
MARKER=PR
for model in ${MODEL[*]}
    do log=$MARKER"_result/"$model"_1000.log"
    nohup python RF_LR_LDA_3.py --model $MODEL --marker $MARKER --repeat 1000 > $log&
    echo 'logging to '$log
done


















