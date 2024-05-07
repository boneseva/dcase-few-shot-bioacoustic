# read csv file and transform it to dcase format
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil


def new_label_row(label, labels, new_row, validation=False):
    # add to dict new_row
    if validation:
        if CONFIG == 2:
            if label.startswith('T') or label.startswith('C'):
                new_row['Q'] = 'POS'
    else:
        if CONFIG == 2 or CONFIG == 3:
            found = False
            for l in labels:
                if l == label:
                    new_row[l] = 'POS'
                    found = True
                else:
                    new_row[l] = 'NEG'
            if not found:
                new_row['OTHER'] = 'POS'
            else:
                new_row['OTHER'] = 'NEG'
        # elif CONFIG == 3:
        #     if label == 'T1':
        #         new_row['T1'] = 'POS'
        #         new_row['C1'] = 'NEG'
        #         new_row['BG'] = 'NEG'
        #         new_row['OTHER'] = 'NEG'
        #     elif label == 'C1':
        #         new_row['T1'] = 'NEG'
        #         new_row['C1'] = 'POS'
        #         new_row['BG'] = 'NEG'
        #         new_row['OTHER'] = 'NEG'
        #     elif label == 'BG':
        #         new_row['T1'] = 'NEG'
        #         new_row['C1'] = 'NEG'
        #         new_row['BG'] = 'POS'
        #         new_row['OTHER'] = 'NEG'
        #     else:
        #         new_row['T1'] = 'NEG'
        #         new_row['C1'] = 'NEG'
        #         new_row['BG'] = 'NEG'
        #         new_row['OTHER'] = 'POS'
        #
        # elif CONFIG == 'all':
        #     for l in labels:
        #         if l == label:
        #             new_row[l] = 'POS'
        #         else:
        #             new_row[l] = 'NEG'

    return new_row


def clean_labels(labels):
    new_labels = []
    for label in labels:
        if (label.startswith('T') or label.startswith('C') or label.startswith('P') or (label.startswith(
                'F') and label!='FLYI') or label == 'BG'):
            new_labels.append(label)
    return new_labels


CONFIG = 3  # 1,2,3,4
TRAIN_RATIO = 0.8
INPUT_FILE = 'all_annotations.nomulti.16k.testing.csv'
TRAIN_NAME = 'train'
VAL_NAME = 'val'
TRAIN_DIR = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data/train'
VAL_DIR = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data/val'
FOLDER = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data'
DATA_DIR = 'C:/Users/Eva/Documents/VibroScape/Annotated recordings nopeaks/16K'

if __name__ == "__main__":

    input_file = INPUT_FILE
    #     empty train and val folders if they exist
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)
    #     make train and val folders
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    # output_file_train = input_file.replace('.csv', 'train.csv')
    # output_file_val = input_file.replace('.csv', 'val.csv')
    #
    # if CONFIG == 'T1_C1_BG_OTHER':
    #     output_file_train = output_file_train.replace('.csv', 't1c1bg.csv')
    #     output_file_val = output_file_val.replace('.csv', 't1c1bg.csv')
    # elif CONFIG == 'all':
    #     output_file_train = output_file_train.replace('.csv', 'all.csv')
    #     output_file_val = output_file_val.replace('.csv', 'all.csv')

    # read csv file
    df = pd.read_csv(input_file, delimiter='\t')

    if CONFIG == 2 or CONFIG == 3:
        # randomly split df rows into train and val
        np.random.seed(42)
        msk = np.random.rand(len(df)) < TRAIN_RATIO
        train_df = df[msk]
        val_df = df[~msk]

        # get all unique labels from column 'label_new'
        labels = list(df['label_new'].unique())
        labels = clean_labels(labels)
        if CONFIG == 2:
            labels_val = ['T', 'C', 'P', 'F']
        elif CONFIG == 3:
            labels_val = labels

        for label in labels_val:
            os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)

        # new dataframe
        new_train_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'] + labels)
        new_val_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime', 'Q'])

        print(f"Processing TRAIN {input_file}...")
        for index, row in tqdm(train_df.iterrows()):
            audio_file = row['audio_file']
            # copy audio file to train folder
            # check if audio file exists
            original_audio_file_path = os.path.join(DATA_DIR, audio_file)
            audio_file_path = os.path.join(TRAIN_DIR, os.path.basename(audio_file))
            if not os.path.exists(audio_file_path):
                shutil.copy(original_audio_file_path, TRAIN_DIR)
            else:
                print(f"File {audio_file_path} already exists.")
            # rename audio file to have the correct path
            # audio_file_name = os.path.join(TRAIN_NAME, os.path.basename(audio_file))
            audio_file_name = os.path.basename(audio_file)
            start_time = row['start_time']
            end_time = row['end_time']
            label = row['label_new']

            new_row = new_label_row(label, labels,
                                    {'Audiofilename': audio_file_name, 'Starttime': start_time, 'Endtime': end_time})

            new_train_df = new_train_df._append(new_row, ignore_index=True)
            # save to new csv file
            # new_train_df.to_csv(output_file_train, index=False)

        print(f"Processing VAL {input_file}...")
        for index, row in tqdm(val_df.iterrows()):
            audio_file = row['audio_file']

            # rename audio file to have the correct path
            audio_file_name = os.path.basename(audio_file)
            start_time = row['start_time']
            end_time = row['end_time']
            label = row['label_new']
            # copy audio file to val folder
            original_audio_file_path = os.path.join(DATA_DIR, audio_file)
            audio_file_path = os.path.join(VAL_DIR, label, os.path.basename(audio_file))
            if not os.path.exists(audio_file_path):
                shutil.copy(original_audio_file_path, os.path.join(VAL_DIR, label))
            else:
                print(f"File {audio_file_path} already exists.")

            new_row = new_label_row(label, labels,
                                    {'Audiofilename': os.path.join(label, audio_file_name), 'Starttime': start_time,
                                     'Endtime': end_time}, validation=True)

            new_val_df = new_val_df._append(new_row, ignore_index=True)
            # save to new csv file
            # new_val_df.to_csv(output_file_val, index=False)

    dfs = [new_train_df, new_val_df]

    for df in dfs:
        # split the data with the same audio file name into the same new csv file
        split = 'train' if df.equals(new_train_df) else 'val'
        df_grouped = df.groupby('Audiofilename')
        for audio_filename, group in df_grouped:
            # print df to csv
            path = FOLDER + '/' + split + '/' + audio_filename
            path = path.replace('.wav', '.csv')
            if split == 'val':
                group['Q'] = 'POS'
                group['Audiofilename'] = audio_filename.split("\\")[-1]
            group.to_csv(path, index=False)

    # remove empty folders inside val folder
    for root, dirs, files in os.walk(VAL_DIR):
        for dir in dirs:
            # delete 'BG', 'BG12', 'OTHER', 'UNKNOWN' and 'UNL' folders
            if (dir in ['BG', 'BG12', 'OTHER', 'UNKNOWN', 'UNL', 'pulc']) or (not os.listdir(os.path.join(root, dir))):
                shutil.rmtree(os.path.join(root, dir))
