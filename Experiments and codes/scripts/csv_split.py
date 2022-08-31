import pandas as pd
import numpy as np


def splitTT(csv_path,att,test_rate,csv_dir,batch_size):
    df = pd.read_csv(csv_path)
    msk = int(test_rate*len(df)) // batch_size * batch_size
    test = df[:msk]
    train = df[msk:]
    train.to_csv(csv_dir+str(att)+'_train.csv', index=False)
    test.to_csv(csv_dir+str(att)+'_test.csv', index=False)


def splitVT(train_csv_path,att,val_rate,csv_dir,batch_size):
    df = pd.read_csv(train_csv_path)
    msk = int(val_rate * len(df)) // batch_size * batch_size
    val = df[:msk]
    train = df[msk:]
    val.to_csv(csv_dir+str(att)+'_val.csv', index=False)
    train.to_csv(csv_dir+str(att)+'_train.csv', index=False)

def splitTVT(csv_path,att,test_rate,val_rate,csv_dir,batch_size):
    splitTT(csv_path, att, test_rate,csv_dir,batch_size)
    splitVT(csv_dir+str(att)+'_train.csv', att, val_rate,csv_dir,batch_size)



def splitAtt(csv_path,csv_dir):
    csv_reader = np.loadtxt(csv_path,delimiter='\t',dtype=str)
    data = {'safe':[],'lively':[],'beautiful':[],'wealthy':[],'depressing':[],'boring':[]}

    study = {'50a68a51fdc9f05596000002': 'safe', '50f62c41a84ea7c5fdd2e454': 'lively',
             '5217c351ad93a7d3e7b07a64': 'beautiful', '50f62cb7a84ea7c5fdd2e458': 'wealthy',
             '50f62ccfa84ea7c5fdd2e459': 'depressing', '50f62c68a84ea7c5fdd2e456': 'boring'}
    for row in csv_reader:
        if (row[1]!='left' and row[1]!='right') or \
                row[4] not in study:
            continue
        data[study[row[4]]].append(row)

    for att in data:
        np.savetxt(csv_dir+str(att)+'.csv', data[att], delimiter='\t',fmt='%s')
