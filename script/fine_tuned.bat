:: This batch file prunes CNN models with given pruning ratio.
:: the commands themselves won’t be printed
:: chcp 65001
@ECHO OFF
ECHO Fine-tuning the model

python main.py --pruned D:/model/best_model0505trial07a_tuned80.pkl --trial trial07a >> trial07a_tuned.txt
