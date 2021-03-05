import os
import cv2
import shutil
import keras
import pandas as pd
import numpy as np
import tensorflow as tf

class DataLoader:
    """
    Data Loader Class
    """

    @staticmethod
    def load_data(data_config, prefix = "blond"):

        generate_dirs(data_config, "Blond_Hair", prefix)

        train_datagen =  keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.inception_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        valid_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.inception_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )


        train_generator = train_datagen.flow_from_directory(
            'data/celeba-dataset/{}-train'.format(prefix),
                target_size=(data_config.data.IMG_HEIGHT, data_config.data.IMG_WIDTH),
                batch_size=data_config.train.BATCH_SIZE)

        validation_generator = valid_datagen.flow_from_directory(
            'data/celeba-dataset/{}-validation'.format(prefix),
                target_size=(data_config.data.IMG_HEIGHT, data_config.data.IMG_WIDTH),
                class_mode='categorical')

        return train_generator, validation_generator

# Helper Functions
"""
Ported from Marcos Alvarado's Repository https://github.com/bmarcoos/udacity-capstone-project
with few changes to the API
"""
def generate_dirs(data_config,attr, prefix):
    """
    partition = {
        0 : train
        1 : validation
        2 : test}
    """
    create_file_folder(prefix)
    print('Generate dataframe with file names for the model')
    df_train, df_val, df_test = generate_df(data_config, attr)
    print('\nCopying the images...')
    copy_images(prefix, attr, df_train, 'train')
    copy_images(prefix, attr, df_val, 'validation')
    copy_images(prefix, attr, df_test, 'test')

def create_file_folder(prefix):
    '''
    create folders in order to use the flow_from_directory method form keras
    
    creates folders:
        train/0
        train/1
        validation/0
        validation/1
        test/0
        test/1
        
    '''
    
    list_type_ds = ['train', 'validation', 'test']
    
    # Delete path if exists
    for typ in list_type_ds:
        if os.path.exists('data/celeba-dataset/{}-{}'.format(prefix, typ)):
            shutil.rmtree('data/celeba-dataset/{}-{}'.format(prefix, typ))
       
    # Create paths for training, validation and test data
    for typ in list_type_ds:
        os.makedirs('data/celeba-dataset/{}-{}'.format(prefix, typ))
        os.makedirs('data/celeba-dataset/{}-{}/0'.format(prefix, typ))
        os.makedirs('data/celeba-dataset/{}-{}/1'.format(prefix, typ))

def generate_df(data_config, attr):
    '''
    select the sub data sets from the recommended partition randomly
    generates balanced data
    
    '''

    df_attr = pd.read_csv(data_config.data.data_folder + 'list_attr_celeba.csv')
    df_attr.set_index('image_id', inplace=True)
    df_attr.replace(to_replace=-1, value=0, inplace=True) # replace -1 by 0
    # Recomended partition
    df_partition = pd.read_csv(data_config.data.data_folder + 'list_eval_partition.csv')

    # join the partition with the attributes
    df_partition.set_index('image_id', inplace=True)
    df_par_attr = df_partition.join(df_attr['Blond_Hair'], how='inner')

    print('Attribute:', attr)

    df_train = df_par_attr[(df_par_attr['partition'] == 0) 
                           & (df_par_attr[attr] == 0)].sample(int(int(data_config.data.TRAINING_SAMPLES)/2))

    df_train = pd.concat([df_train,
                      df_par_attr[(df_par_attr['partition'] == 0) 
                                  & (df_par_attr[attr] == 1)].sample(int(int(data_config.data.TRAINING_SAMPLES)/2))])

    df_val = df_par_attr[(df_par_attr['partition'] == 1) 
                            & (df_par_attr[attr] == 0)].sample(int(int(data_config.data.VALIDATION_SAMPLES)/2)) #file names for validation

    df_val = pd.concat([df_val,
                        df_par_attr[(df_par_attr['partition'] == 1) & (df_par_attr[attr] == 1)].sample(int(int(data_config.data.VALIDATION_SAMPLES)/2))]) #file names for validation

    df_test = df_par_attr[(df_par_attr['partition'] == 2) & (df_par_attr[attr] == 0)].sample(int(int(data_config.data.TEST_SAMPLES)/2)) 
    #file names for test
    df_test = pd.concat([df_test,
                                  df_par_attr[(df_par_attr['partition'] == 2) & (df_par_attr[attr] == 1)].sample(int(int(data_config.data.TEST_SAMPLES)/2))]) #file names for test

    return df_train, df_val, df_test

def copy_images(folder_prefix, attribute, df_images, df_type):
    '''
    copy images to the corresponding folder (classes)
    
    folder_prefix: expected prefix in folder
    attribute: attribute as in the data set to discriminate the classes
    df_images: data frame with image file name and attributes
    df_type: type of data set, train, validation or test
    
    '''
    
    # Copy images
    for i, j in df_images.iterrows():
        if j[attribute] == 0:
            shutil.copy('data/CelebA/img_align_celeba/img_align_celeba/' + i, 'data/celeba-dataset/{}-{}/0/{}'.format(folder_prefix, df_type, i))
        if j[attribute] == 1:
            shutil.copy('data/CelebA/img_align_celeba/img_align_celeba/' + i, 'data/celeba-dataset/{}-{}/1/{}'.format(folder_prefix, df_type, i))
            
    print("{} {} - Copy Images: DONE!".format(folder_prefix, df_type))