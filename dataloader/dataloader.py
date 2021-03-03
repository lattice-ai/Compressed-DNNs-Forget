import cv2
import keras
import pandas as pd
import numpy as np

class DataLoader:
    """
    Data Loader Class
    """

    @staticmethod
    def get_train_data(data_config):
        
        x_train, y_train = generate_df(data_config, 0, "Blond_Hair", data_config.data.TRAINING_SAMPLES)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.inception_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        train_datagen.fit(x_train)

        train_generator = train_datagen.flow(x_train, y_train,
            batch_size= data_config.train.batch_size)

        return train_generator


## Helper Functions

def load_reshape_img(fname):
    img = keras.preprocessing.image.load_img(fname)
    x = keras.preprocessing.image.img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)
    return x

def generate_df(data_config,partition, attr, num_samples):
    """
    partition = {
        0 : train
        1 : validation
        2 : test}
    """

    df_attr = pd.read_csv(data_config.data.data_folder + 'list_attr_celeba.csv')
    df_attr.set_index('image_id', inplace=True)
    df_attr.replace(to_replace=-1, value=0, inplace=True) # replace -1 by 0
    # Recomended partition
    df_partition = pd.read_csv(data_config.data.data_folder + 'list_eval_partition.csv')

    # join the partition with the attributes
    df_partition.set_index('image_id', inplace=True)
    df_par_attr = df_partition.join(df_attr['Blond_Hair'], how='inner')

    
    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(int(num_samples)/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(int(num_samples)/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(data_config.data.images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = keras.utils.np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(data_config.data.images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (data_config.data.IMG_WIDTH, data_config.data.IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_