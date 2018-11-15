from __future__ import print_function
import os.path as osp
import sys
import cv2
import pandas as pd
import numpy as np

# The data sets to be downloaded
d_sets = ['yt_bb_detection_validation', 'yt_bb_detection_train']

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']


def make_filename(df_row):
    return 'download/yt_bb_detection_validation/' + str(df_row['class_id']) + '/' + \
            df_row['youtube_id'] + "_" + str(df_row['timestamp_ms']) + '_' + \
            str(df_row['class_id']) + '_' + str(df_row['object_id']) + '.jpg'


def convert_coord(df_row, width, height):
    x1, x2, y1, y2 = df_row.values[6:10]
    x1 = int(x1*width)
    x2 = int(x2*width)
    y1 = int(y1*height)
    y2 = int(y2*height)
    return x1, x2, y1, y2


def parse_video(data, vid, f):
    print('Processing', vid, '...')

    # retrieve video resolution (iteratre till success)
    for index, row in data.iterrows():
        fname = make_filename(row)
        if osp.exists(fname) and osp.getsize(fname) > 10240:
            print(fname)
            img = cv2.imread(fname)
            height, width = img.shape[:2]
            print("Resolution:", img.shape[:2])
            break

    # iterate through the labeled frames
    for i in range(data.shape[0]-1):
        if data.iloc[i]['object_presence'] != 'present' or data.iloc[i+1]['object_presence'] != 'present':
            continue
        if data.iloc[i+1]['timestamp_ms'] - data.iloc[i]['timestamp_ms'] > 1000:
            # print("Not continuous!")
            continue
        img1_filename = make_filename(data.iloc[i])
        img2_filename = make_filename(data.iloc[i+1])
        # check file existence
        if not osp.exists(img1_filename) or not osp.exists(img2_filename) or \
            osp.getsize(img1_filename) < 10240 or osp.getsize(img2_filename) < 10240:
            continue
        x1_1, x1_2, y1_1, y1_2 = convert_coord(data.iloc[i], width, height)
        x2_1, x2_2, y2_1, y2_2 = convert_coord(data.iloc[i+1], width, height)
        f.write('%s,%s,%d,%d,%d,%d,%d,%d,%d,%d\n' % 
                (img1_filename, img2_filename, x1_1, y1_1, x1_2-x1_1, y1_2-y1_1, x2_1, y2_1, x2_2-x2_1, y2_2-y2_1))


def parse_all():
    df = pd.read_csv('yt_bb_detection_validation.csv', header=None, index_col=False)
    df.columns = col_names

    # Get list of unique video files
    vids = df['youtube_id'].unique()

    with open('imglist.txt', 'w') as f:
        for vid in vids[100:1100]:
            parse_video(df[df['youtube_id'] == vid], vid, f)


if __name__ == "__main__":
    parse_all()
