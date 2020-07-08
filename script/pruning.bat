:: This batch file prunes CNN models with given pruning ratio.
:: the commands themselves won’t be printed
ECHO Pruning the model



SET "trial=trial07c"
SET "model=D:/model/bestmodel0525_trial07c.pkl"
REM SET "model1=D:/model/bestmodel0604_trial07b5_pruned20.pkl"
REM SET "model2=D:/model/bestmodel0604_trial07b5_pruned40.pkl"
REM SET "model3=D:/model/bestmodel0604_trial07b5_pruned60.pkl"
REM SET "model4=D:/model/bestmodel0604_trial07b5_pruned80.pkl"

SET "log=trial07c.txt"
python prune.py --model %model% --trial %trial% --percent 20 --plot

REM python prune.py --model %model% --trial %trial% --percent 20  >> %log%
REM python main.py --pruned %model1% --trial %trial% --epoch 30 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 10  --lr 5e-02 >> %log%


REM python prune.py --model %model% --trial %trial% --percent 40 >> %log%
REM python main.py --pruned %model2% --trial %trial% --epoch 80 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 10  --lr 1e-01 >> %log%

REM python prune.py --model %model% --trial %trial% --percent 60 >> %log%
REM python main.py --pruned %model3% --trial %trial% --epoch 150 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 10  --lr 1e-01 >> %log%

REM python prune.py --model %model% --trial %trial% --percent 80 >> %log%
REM python main.py --pruned %model4% --trial %trial% --epoch 300 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 15  --lr 5e-01 >> %log%


