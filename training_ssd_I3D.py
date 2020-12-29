


from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_I3D_FFssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_3d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms




load_prev_model=0
initial_learning_rate=0.0001#0.0001
batch_size = 8 # Change the batch size if you like, or if you run into memory issues with your GPU.
epochs = 270
img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images
mean_color = None# [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = False#True # [2, 1, 0]The color channel order in the original SSD is BGR
n_classes = 4 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_coco
aspect_ratios = [[1.0,0.1,10, 2.0, 0.5],
                 [1.0,0.1,10, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0,0.1,10, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = None #[8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

input_type = 'liteflow'  # 'TVL1','RGB'

images_dir = '.'
if input_type == 'liteflow':

    train_labels_filename = './data/training_liteflow.csv'
    val_labels_filename = './data/val_liteflow.csv'
elif input_type == 'RGB':
    train_labels_filename = './data/training_rgb.csv'
    val_labels_filename = './data/val_rgb.csv'
elif input_type == 'TVL1':
    train_labels_filename = './data/training_Tvl1.csv'
    val_labels_filename = './data/val_Tvl1.csv'

# ## 2. Build or load the model

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.
num_frame=10
model, predictor_sizes = ssd_300(input_shape=(num_frame,img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.00004,#0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)



# 2: Load the trained VGG-16 weights into the model.


weights_path = './weights/i3d/i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
model.load_weights(weights_path, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
if load_prev_model==1:
    if input_type == 'liteflow':

        model.load_weights('./weights/i3d/FFssd300_I3D_liteflow_final.h5', by_name=True)
    elif input_type == 'RGB':
        model.load_weights('./weights/i3d/FFssd300_I3d_final.h5', by_name=True)
    elif input_type == 'TVL1':
        model.load_weights('./weights/i3d/FFssd300_I3D_Tvl1_final.h5', by_name=True)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)






# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.



train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)



# 2: Parse the image and label lists for the training and validation datasets.


# Images

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')


# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


resize = Resize(height=img_height, width=img_width)


# 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('fusion_conv4_b_3_c_mbox_conf').output_shape[1:3],
                   model.get_layer('Mixed_4c_mbox_conf').output_shape[1:3],
                   model.get_layer('Mixed_5c_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.05,
                                    normalize_coords=normalize_coords)




# 5: Set the image processing / data augmentation options and create generator handles.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         num_frame=num_frame,
                                         shuffle=False,
                                         transformations=[resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     num_frame=num_frame,
                                     shuffle=False,
                                     transformations=[resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# ## 4. Run the training



# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch <= 100: return 0.001
    else: return 0.0001


# TODO: Set the number of epochs to train for.
my_log_dir='./logs'
results_dir='./results'


model_checkpoint = ModelCheckpoint(filepath='FFssd_i3d_'+input_type+'_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd300_I3D_pascal_07+12_training_log_'+input_type+'.csv',
                       separator=',',
                       append=True)


learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss',factor=0.75,patience=max(int(epochs/30),3),verbose = 1, epsilon=1e-3,min_lr=1e-5)#LearningRateScheduler(schedule=lr_schedule, verbose=1)

terminate_on_nan = TerminateOnNaN()
Tensor_view=TensorBoard(log_dir=my_log_dir,histogram_freq=0, write_graph=True,write_images=True)

callbacks_list = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
			 Tensor_view,
             terminate_on_nan]


if load_prev_model==0:
    history = model.fit_generator(generator = train_generator,
                                  steps_per_epoch = ceil(train_dataset_size/batch_size),
                                  epochs = epochs,
                                  callbacks =  callbacks_list,
                                  validation_data = val_generator,
                                  validation_steps = ceil(val_dataset_size/batch_size))


    model_name = './weights/i3d/FFssd300_I3D_'+input_type+'_final'
    model.save('{}.h5'.format(model_name))
    model.save_weights('{}_weights.h5'.format(model_name))

    print()
    print("Model saved under {}.h5".format(model_name))
    print("Weights also saved separately under {}_weights.h5".format(model_name))
    print()


# ## 5. Make predictions


### Make predictions

# 1: Set the generator

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         num_frame=num_frame,
                                         transformations=[resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)


# 2: Generate samples

X, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))



# 3: Make a prediction


y_pred = model.predict(X)



# 4: Decode the raw prediction `y_pred`

y_pred_decoded = decode_detections(y_pred,
                          confidence_thresh=0.5,
                          iou_threshold=0.2,
                          top_k=200,
                          normalize_coords=normalize_coords,
                          img_height=img_height,
                          img_width=img_width)

# 5: Convert the predictions for th

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])


# 5: Draw the predicted boxes onto the image

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['background', 'human_walking', 'human_crawl', 'human_running', 'human_walkingCrouched']

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i][i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()