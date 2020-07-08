:: This batch file prunes CNN models with given pruning ratio.
:: the commands themselves won’t be printed
REM ECHO OFF
ECHO Soft pruning the model

SET "model=D:/model/bestmodel0525_trial07c.pkl"

SET "trial=trial07cs"
SET "log=trial07cs.txt"


REM python soft_prune.py --resume %model% --trial %trial% --epoch 50 --patience 10  --lr 1e-02 --percent 20 >> %log%
REM python soft_prune.py --resume %model% --trial %trial% --epoch 100 --patience 10  --lr 1e-02 --percent 40 >> %log%
python soft_prune.py --resume %model% --trial %trial% --epoch 150 --patience 10  --lr 1e-02 --percent 60 --freq 3 >> %log%
REM python soft_prune.py --resume %model% --trial %trial% --epoch 200 --patience 10  --lr 1e-02 --percent 80 >> %log%
