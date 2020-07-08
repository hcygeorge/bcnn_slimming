:: This batch file prunes CNN models with given pruning ratio.
@ECHO OFF
ECHO %date% %time%
ECHO %CD%
ECHO Testing the model
REM ECHO %errorlevel%


SET "model=D:/model/bestmodel0511_trial07aiter.pkl"
python test.py --resume %model%




