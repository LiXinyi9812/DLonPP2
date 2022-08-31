import os

def rename(path):
    for cur_path, dirs, files in os.walk(path):
        for file_name in files:
            if '_' in file_name:
                newname = file_name.split('_')[2]
                os.rename(path+'/'+file_name, path+'/'+newname+'.JPG')
