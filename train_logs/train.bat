:: This batch file prunes CNN models with given pruning ratio.
:: the commands themselves won’t be printed
REM ECHO OFF
ECHO Train the model

SET "model="

SET "trial=trial07b3"
SET "log=trial07b3.txt"

SET "model=D:/model/bestmodel0525_trial07b3_pruned20.pkl"

python main.py --pruned %model% --trial %trial% --epoch 100 --patience 10   --sparsity 1e-04 --first 1e-04 --last 1e-03 --lr 5e-02 >> %log%
