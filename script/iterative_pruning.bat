:: This batch file prunes CNN models with given pruning ratio.
REM @ECHO OFF
ECHO %date% %time%
ECHO %CD%
ECHO Pruning the model

SET "trial=trial07citer"
SET "model=D:/model/bestmodel0525_trial07c.pkl"
SET "model1=D:/model/bestmodel0525_trial07c_pruned20.pkl"
SET "model2=D:/model/bestmodel0602_trial07citer_pruned25.pkl"
SET "model3=D:/model/bestmodel0602_trial07citer_pruned33.pkl"
SET "model4=D:/model/bestmodel0602_trial07citer_pruned50.pkl"
SET "log=trial07citer.txt"

REM python prune.py --model %model% --trial %trial% --percent 20 >> %log%
REM python main.py --pruned %model1% --trial %trial% --epoch 100 >> %log%

python prune.py --model %model1% --trial %trial% --percent 25 >> %log%
python main.py --pruned %model2% --trial %trial% --epoch 25 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 5  --lr 1e-02>> %log%

python prune.py --model %model2% --trial %trial% --percent 33 >> %log%
python main.py --pruned %model3% --trial %trial% --epoch 100 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 10  --lr 1e-02>> %log%

python prune.py --model %model3% --trial %trial% --percent 50 >> %log%
python main.py --pruned %model4% --trial %trial% --epoch 200 --sparsity 1e-04 --first 1e-04 --last 1e-04 --patience 10  --lr 1e-02>> %log%