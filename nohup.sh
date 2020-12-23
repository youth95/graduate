#conda deactivate
source venv/bin/activate
nohup python -u ./tasks/image_train.py >> run.log &