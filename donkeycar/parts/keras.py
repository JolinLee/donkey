""""

keras.py

functions to run and train autopilots using keras

"""

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from donkeycar import util


class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist


class KerasCategorical(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical()  

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        angle_unbinned = util.data.linear_unbin(angle_binned[0])
        return angle_unbinned, throttle[0][0]


class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_n_linear(num_outputs)
        else:
            self.model = default_linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_categorical():
    img_in = Input(shape=(120, 160, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(
        x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .01})

    return model


from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.models import Model

# 新Model mobilenet
def default_mobilenet():
    img_rows, img_cols, img_channel = 120,160 , 3 
    
    #--------
    base_model = MobileNet (weights=None , include_top=False,
                         input_shape=(img_rows, img_cols, img_channel))
    #---------
    
    
    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.5)(x)
    
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)        
    
    model = Model(inputs=base_model.input, outputs=[angle_out,throttle_out])                 

    model.compile(loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
              optimizer='adam', metrics=['accuracy'])
    return model

from tensorflow.python.keras import backend as K


def linear_unbin_layer(tnsr):
    bin = K.constant((2 / 14), dtype='float32')
    norm = K.constant(1, dtype='float32')

    b = K.cast(K.argmax(tnsr), dtype='float32')
    a = b - norm
    # print('linear_unbin_layer out: {}'.format(a))
    return a


def default_catlin():
    """
    Categorial Steering output before linear conversion.
    :return:
    """
    img_in = Input(shape=(120, 160, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_cat_out = Dense(15, activation='softmax', name='angle_cat_out')(x)
    angle_out = Dense(1, activation='sigmoid', name='angle_out')(angle_cat_out)
    # angle_out = Lambda(linear_unbin_layer, output_shape=(1,1, ), name='angle_out')(angle_cat_out)

    # continuous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .01})

    return model


def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='linear')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='linear')(x)
    x = Dropout(.1)(x)
    # categorical output of the angle
    angle_out = Dense(1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model


def default_n_linear(num_outputs):
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Cropping2D(cropping=((60, 0), (0, 0)))(x)  # trim 60 pixels off top
    x = Lambda(lambda x: x / 127.5 - 1.)(x)  # normalize and re-center
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs)

    model.compile(optimizer='adam',
                  loss='mse')

    return model


class KerasRNN_LSTM(KerasPilot):
    def __init__(self, image_w =160, image_h=120, image_d=3, seq_length=3, num_outputs=2, *args, **kwargs):
        super(KerasRNN_LSTM, self).__init__(*args, **kwargs)
        image_shape = (image_h, image_w, image_d)
        self.model = rnn_lstm(seq_length=seq_length,
            num_outputs=num_outputs,
            image_shape=image_shape)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()
        self.optimizer = "rmsprop"

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                  loss='mse')

    def run(self, img_arr):
        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        
        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w, self.image_d )
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle
  

def rnn_lstm(seq_length=3, num_outputs=2, image_shape=(120,160,3)):

    from keras.layers import Input, Dense
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization, Merge
    from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
    from keras.layers.merge import concatenate
    from keras.layers import LSTM
    from keras.layers.wrappers import TimeDistributed as TD

    img_seq_shape = (seq_length,) + image_shape   
    img_in = Input(batch_shape = img_seq_shape, name='img_in')
    
    x = Sequential()
    x.add(TD(Cropping2D(cropping=((60,0), (0,0))), input_shape=img_seq_shape )) #trim 60 pixels off top
    x.add(TD(Convolution2D(24, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (3,3), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(MaxPooling2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(.1)))
      
    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_out"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))
    
    return x


class Keras3D_CNN(KerasPilot):
    def __init__(self, image_w =160, image_h=120, image_d=3, seq_length=20, num_outputs=2, *args, **kwargs):
        super(Keras3D_CNN, self).__init__(*args, **kwargs)
        self.model = build_3d_cnn(w=image_w, h=image_h, d=image_d, s=seq_length, num_outputs=num_outputs)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

    def run(self, img_arr):

        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        
        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w, self.image_d )
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle


def build_3d_cnn(w, h, d, s, num_outputs):
    from keras.layers import Input, Dense
    from keras.models import Sequential
    from keras.layers import Conv3D, MaxPooling3D, Reshape, BatchNormalization, Merge
    from keras.layers import Activation, Dropout, Flatten, Cropping3D

    #Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py
    '''
        w : width
        h : height
        d : depth
        s : n_stacked
    '''
    input_shape=(s, h, w, d)

    model = Sequential()
    #First layer
    model.add(Cropping3D(cropping=((0,0), (50,10), (0,0)), input_shape=input_shape) ) #trim pixels off top
    
    # Second layer
    model.add(Conv3D(
        filters=16, kernel_size=(3,3,3), strides=(1,3,3),
        data_format='channels_last', border_mode='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Third layer
    model.add(Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', border_mode='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fourth layer
    model.add(Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', border_mode='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fifth layer
    model.add(Conv3D(
        filters=128, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', border_mode='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fully connected layer
    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    #model.add(Activation('tanh'))

    return model