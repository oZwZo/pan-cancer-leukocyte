






for i in {1..2}  
do  
name='S_'$i
Rlog='SVM_rbf_'$i'_100.log'	
Llog='SVM_linear_'$i'_100.log'
#echo $name

#echo $log
nohup python SVM.py --title $name --kernel linear --repeat 100 > result/SVM_result/$Rlog &
nohup python SVM.py --title $name --kernel rbf --repeat 100  > result/SVM_result/$Llog &
done  






