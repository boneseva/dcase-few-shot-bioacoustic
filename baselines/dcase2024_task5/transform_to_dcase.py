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
            if label.startswith('T') or label.startswith('C') or label.startswith('P') or label.startswith('F'):
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
                new_row['OTHER'] = label
            else:
                new_row['OTHER'] = 'NEG'

    return new_row


def clean_labels(labels):
    new_labels = []
    for label in labels:
        if (label.startswith('T') or label.startswith('C') or label.startswith('P') or (label.startswith(
                'F') and label != 'FLYI') or label == 'BG'):
            new_labels.append(label)
    return new_labels


CONFIG = 2  # 1,2,3,4
TRAIN_RATIO = 0.8
TRAIN_NAME = 'train'
VAL_NAME = 'val'
TRAIN_DIR = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data_testing/train'
VAL_DIR = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data_testing/val'
TEST_DIR = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data_testing/test'
FOLDER = 'C:/Users/Eva/Documents/VibroScape/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/data_testing'
DATA_DIR = r'C:\Users\Eva\Documents\VibroScape\dataset\subset\train'
TEST_DATA_DIR = r'C:\Users\Eva\Documents\VibroScape\dataset\subset\test'
INPUT_FILE = DATA_DIR + '/train.csv'
INPUT_TEST_FILE = TEST_DATA_DIR + '/test.csv'
TRAIN = True
TEST = True

TESTING = False
SCRATCH = True

if __name__ == "__main__":

    if TESTING:
        input_file = r'C:\Users\Eva\Documents\VibroScape\dataset\all_annotations.nomulti.16k.testing.csv'
        TRAIN_DIR = TRAIN_DIR + '_testing'
        VAL_DIR = VAL_DIR + '_testing'
        TEST_DIR = TEST_DIR + '_testing'
    else:
        input_file = INPUT_FILE
    #     empty train and val folders if they exist

    if SCRATCH:
        if os.path.exists(TRAIN_DIR):
            shutil.rmtree(TRAIN_DIR)
        if os.path.exists(VAL_DIR):
            shutil.rmtree(VAL_DIR)
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
    #     make train and val folders
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # read csv file
    df = pd.read_csv(input_file, delimiter='\t')

    if CONFIG == 2 or CONFIG == 3:
        # get all unique labels from column 'label_new'
        if CONFIG == 2:
            labels_val = ['T', 'C', 'P', 'F']
            labels = ['T', 'C', 'P', 'F', 'BG']
        elif CONFIG == 3:
            labels = list(df['label_new'].unique())
            labels = clean_labels(labels)
            labels_val = labels

        for label in labels_val:
            os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)

        new_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'] + labels)

        if TRAIN:
            print(f"Processing {input_file}...")
            for index, row in tqdm(df.iterrows()):
                audio_file = row['audio_file']
                # copy audio file to train folder
                # check if audio file exists
                original_audio_file_path = os.path.join(DATA_DIR, audio_file)
                audio_file_path = os.path.join(TRAIN_DIR, os.path.basename(audio_file))
                if not os.path.exists(audio_file_path):
                    shutil.copy(original_audio_file_path, TRAIN_DIR)
                # else:
                #     print(f"File {audio_file_path} already exists.")
                # rename audio file to have the correct path
                # audio_file_name = os.path.join(TRAIN_NAME, os.path.basename(audio_file))
                audio_file_name = os.path.basename(audio_file)
                start_time = row['start_time']
                end_time = row['end_time']
                label = row['label_new']

                if CONFIG == 2:
                    if not label.startswith('T') and not label.startswith('C') and not label.startswith(
                            'P') and not label.startswith('F') and label != 'BG':
                        print(f"---------- Skipping {label} for {audio_file_name}")
                        continue
                    else:
                        print(f"Processing {label} for {audio_file_name}")
                        classification = label[0] if label != 'BG' else 'BG'

                    new_row = new_label_row(classification, labels,
                                            {'Audiofilename': audio_file_name, 'Starttime': start_time,
                                             'Endtime': end_time})

                new_df = new_df._append(new_row, ignore_index=True)
                # save to new csv file
                # new_train_df.to_csv(output_file_train, index=False)

            print(f"Splitting into TRAIN and VAL with ratio {TRAIN_RATIO}...")
            # randomly split df rows into train and val
            np.random.seed(42)
            msk = np.random.rand(len(new_df)) < TRAIN_RATIO
            new_train_df = new_df[msk]
            val_df = new_df[~msk]

            print(f"Processing VAL {input_file}...")
            new_val_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime', 'Q'])
            for index, row in tqdm(val_df.iterrows()):
                audio_file = row['Audiofilename']

                # rename audio file to have the correct path
                audio_file_name = os.path.basename(audio_file)
                start_time = row['Starttime']
                end_time = row['Endtime']

                # print(f'Row: {row}')
                # find in which column the key is 'POS'
                label = row.loc[row == 'POS'].index[0]

                # copy audio file to val folder
                original_audio_file_path = os.path.join(TRAIN_DIR, audio_file)
                audio_file_path = os.path.join(VAL_DIR, label, os.path.basename(audio_file))
                if os.path.exists(os.path.join(VAL_DIR, label)):
                    if not os.path.exists(audio_file_path):
                        shutil.copy(original_audio_file_path, os.path.join(VAL_DIR, label))
                    # else:
                    #     print(f"File {audio_file_path} already exists.")

                    new_row = {'Audiofilename': f"{label}\\{audio_file_name}", 'Starttime': start_time, 'Endtime': end_time,
                               'Q': 'POS'}

                    new_val_df = new_val_df._append(new_row, ignore_index=True)
                # save to new csv file
                # new_val_df.to_csv(output_file_val, index=False)

    test_df = pd.read_csv(INPUT_TEST_FILE, delimiter='\t')
    new_test_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime', 'Q'])

    if TEST:
        print(f"Processing TEST {INPUT_TEST_FILE}...")
        if CONFIG == 2:
            labels_test = ['T', 'C', 'P', 'F']
            labels = ['T', 'C', 'P', 'F', 'BG']
        elif CONFIG == 3:
            labels = list(df['label_new'].unique())
            labels = clean_labels(labels)
            labels_test = labels

        for label in labels_test:
            os.makedirs(os.path.join(TEST_DIR, label), exist_ok=True)

        for index, row in tqdm(test_df.iterrows()):
            audio_file = row['audio_file']

            # rename audio file to have the correct path
            audio_file_name = os.path.basename(audio_file)
            start_time = row['start_time']
            end_time = row['end_time']
            label = row['label_new']

            if CONFIG == 2:
                if not label.startswith('T') and not label.startswith('C') and not label.startswith(
                        'P') and not label.startswith('F'):
                    print(f"---------- Skipping {label} for {audio_file_name}")
                    continue
                else:
                    print(f"Processing {label} for {audio_file_name}")
                    classification = label[0]

            # copy audio file to val folder
            original_audio_file_path = os.path.join(TEST_DATA_DIR, audio_file)
            audio_file_path = os.path.join(TEST_DIR, classification, os.path.basename(audio_file))
            if os.path.exists(os.path.join(TEST_DIR, classification)):
                if not os.path.exists(audio_file_path):
                    shutil.copy(original_audio_file_path, os.path.join(TEST_DIR, classification))
                # else:
                #     print(f"File {audio_file_path} already exists.")

                new_row = {'Audiofilename': f"{classification}\\{audio_file_name}", 'Starttime': start_time, 'Endtime': end_time,
                           'Q': 'POS'}

                new_test_df = new_test_df._append(new_row, ignore_index=True)
            # save to new csv file
            # new_val_df.to_csv(output_file_val, index=False)

    dfs = [new_train_df, new_val_df, new_test_df]

    for df in dfs:
        # split the data with the same audio file name into the same new csv file
        if df.equals(new_train_df):
            split = 'train'
        elif df.equals(new_val_df):
            split = 'val'
        else:
            split = 'test'
        if TESTING:
            split_folder = split + '_testing'
        else:
            split_folder = split
        df_grouped = df.groupby('Audiofilename')
        for audio_filename, group in df_grouped:
            # print df to csv
            path = FOLDER + '/' + split_folder + '/' + audio_filename
            path = path.replace('.wav', '.csv')
            if split == 'val' or split == 'test':
                group['Q'] = 'POS'
                group['Audiofilename'] = audio_filename.split("\\")[-1]
            group.to_csv(path, index=False)

    # remove empty folders inside val folder
    for root, dirs, files in os.walk(VAL_DIR):
        for dir in dirs:
            # delete 'BG', 'BG12', 'OTHER', 'UNKNOWN' and 'UNL' folders
            if (dir in ['BG', 'BG12', 'OTHER', 'UNKNOWN', 'UNL', 'pulc']) or (not os.listdir(os.path.join(root, dir))):
                shutil.rmtree(os.path.join(root, dir))

    for root, dirs, files in os.walk(TEST_DIR):
        for dir in dirs:
            # delete 'BG', 'BG12', 'OTHER', 'UNKNOWN' and 'UNL' folders
            if (dir in ['BG', 'BG12', 'OTHER', 'UNKNOWN', 'UNL', 'pulc']) or (not os.listdir(os.path.join(root, dir))):
                shutil.rmtree(os.path.join(root, dir))
