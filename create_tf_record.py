"""
    converts KITTI object detection dataset to tfrecord for use
    with TF object detection API for vehicle detection
    can be used for singleclass or multiclass data
"""

import tensorflow as tf
import dataset_util # this is found in /research/object_detection/utils/ (Tensorflow Object Detection API)
import os.path
from PIL import Image

__author__ = "Moritz Kampelmuehler"

# constants
TESTSPLIT = 10 # samples (leave TESTSPLIT samples for test split)
NUM_SAMPLES = 7481
IMAGE_FORMAT = b'png'
BASEDIR = 'training/' # specify the basedir
IMGDIR = 'image_2/'
LABDIR = 'label_2/'
MODE = 'car_only' # for single class detector
# MODE = 'multi_class' # for multi class detector (Car (1), Van (2), Truck(3))
VEHICLE_LABELS = ['Car', 'Van', 'Truck']
VEHICLE_LABEL_IDS = {'Car': 1, 'Van': 2, 'Truck': 3}
SHUFFLE = True

def create_tf_example(height, width, filename, encoded_image_data, image_format, xmins, xmaxs, ymins, ymaxs, classes_text, classes):
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height), # Image height
      'image/width': dataset_util.int64_feature(width), # Image width
      'image/filename': dataset_util.bytes_feature(filename), # Filename of the image
      'image/source_id': dataset_util.bytes_feature(filename), # Filename of the image
      'image/encoded': dataset_util.bytes_feature(encoded_image_data), # Encoded image bytes
      'image/format': dataset_util.bytes_feature(image_format), # b'jpeg' or b'png'
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins), # normalized left x coordinate in bounding box
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs), # normalized right x coordinate in bounding box
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins), # normalized top y coordinate in bounding box
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs), # normalized bottom y coordinate in bounding box
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text), # string class name of bounding box
      'image/object/class/label': dataset_util.int64_list_feature(classes), # integer class id of bounding box
  }))
  return tf_example
  
def loadAnnotations():
  annotations = []
  for i in range(NUM_SAMPLES):
    annotation = {'file_name': '{0:06d}.png'.format(i)}
    with open(os.path.join(BASEDIR, LABDIR, '{0:06d}.txt'.format(i)), 'r') as f:
      bboxes = []
      for line in f:
        bbox = {}
        line_split = line.split(' ')
        
        if line_split[0] not in VEHICLE_LABELS:
          # sort out non-vehicle entries
          continue
        
        # create bounding box
        bbox['left'] = float(line_split[4])
        bbox['right'] = float(line_split[6])
        bbox['top'] = float(line_split[5])
        bbox['bottom'] = float(line_split[7])
        
        if MODE == 'car_only':
          bbox['label'] = 'Car'
          bbox['label_id'] = VEHICLE_LABEL_IDS['Car']
        elif MODE == 'multi_class':
          bbox['label'] = line_split[0]
          bbox['label_id'] = VEHICLE_LABEL_IDS[line_split[0]] 
        else:
          raise ValueError('unknown MODE')
        bboxes.append(bbox) 
             
      if not bboxes:
        # sort out non-vehicle frames
        continue
      annotation['bbox'] = bboxes
    annotations.append(annotation)
  print('{} {} annotations of {} total annotations loaded succesfully'.format(len(annotations), MODE, NUM_SAMPLES))
  return annotations

def createTFRecord(mode, annotations):
  writer = tf.python_io.TFRecordWriter('KITTI_vehicle_{}.tfrecord'.format(mode))  
  if mode == 'train':
    sample_range = range(len(annotations)-TESTSPLIT)
  elif mode == 'test':
    sample_range = range(-TESTSPLIT,0)
  else:
    raise ValueError('unknown mode')
    
  for n in sample_range:
    print('Processing file {0:06d} of {1:06d}'.format(n+1 if mode == 'train' else n+1+TESTSPLIT, len(sample_range)))
    filename = annotations[n]['file_name']
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = ([] for i in range(6))
    # read image
    image_location = os.path.join(BASEDIR, IMGDIR, filename)
    with tf.gfile.GFile(image_location, 'rb') as fid:
      encoded_image_data = fid.read()
    # get image size
    im = Image.open(image_location)
    width, height = im.size
    for annotation in annotations[n]['bbox']:
      xmins += [annotation['left']/width]
      xmaxs += [annotation['right']/width]
      ymins += [annotation['top']/height]
      ymaxs += [annotation['bottom']/height]
      classes_text += [annotation['label'].encode('utf8')]
      classes += [annotation['label_id']]
    tf_example = create_tf_example(height, width, filename.encode('utf8'), encoded_image_data, IMAGE_FORMAT, xmins, xmaxs, ymins, ymaxs, classes_text, classes)
    writer.write(tf_example.SerializeToString())
  writer.close()
  
def main(_):
  annotations = loadAnnotations()
  if SHUFFLE:
    from random import shuffle
    shuffle(annotations)
    
  createTFRecord('train', annotations)
  createTFRecord('test', annotations)

if __name__ == '__main__':
  tf.app.run()
