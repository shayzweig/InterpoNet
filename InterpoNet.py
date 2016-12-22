import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import argparse

# Parsing the parameters
parser = argparse.ArgumentParser(description='Interponet inference')
parser.add_argument('img1_filename', type=str, help='First image filename in the image pair')
parser.add_argument('img2_filename', type=str, help='Second image filename in the image pair')
parser.add_argument('edges_filename', type=str, help='Edges filename')
parser.add_argument('matches_filename', type=str, help='Sparse matches filename')
parser.add_argument('out_filename', type=str, help='Flow output file filename')

parser.add_argument('--model_filename', type=str, help='Saved model parameters filename')
parser.add_argument('--ba_matches_filename', type=str, help='Sparse matches filename from Second image to first image')

parser.add_argument('--img_width', type=int, help='Saved model parameters filename')
parser.add_argument('--img_height', type=int, help='Saved model parameters filename')
parser.add_argument('--downscale', type=int, help='Saved model parameters filename')

parser.add_argument('--sintel', action='store_true', help='Use default parameters for sintel')

args = parser.parse_args()

if args.sintel:
    if args.img_width is None:
        args.img_width = 1024
    if args.img_height is None:
        args.img_height = 436
    if args.downscale is None:
        args.downscale = 8
    if args.model_filename is None:
        args.model_filename = 'models/ff_sintel.ckpt'

# Load edges file
print "Loading files..."
edges = io_utils.load_edges_file(args.edges_filename, width=args.img_width, height=args.img_height)

# Load matching file
img, mask = io_utils.load_matching_file(args.matches_filename, width=args.img_width, height=args.img_height)

# downscale
print "Downscaling..."
img, mask, edges = utils.downscale_all(img, mask, edges, args.downscale)

if args.ba_matches_filename is not None:
    img_ba, mask_ba = io_utils.load_matching_file(args.ba_matches_filename, width=args.img_width, height=args.img_height)

    # downscale ba
    img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
    img, mask =  utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)


with tf.device('/gpu:0'):
    with tf.Graph().as_default():

        image_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 2), name='image_ph')
        mask_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='mask_ph')
        edges_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='edges_ph')

        forward_model = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)

        saver_keep = tf.train.Saver(tf.all_variables(), max_to_keep=0)

        sess = tf.Session()

        saver_keep.restore(sess, args.model_filename)

        print "Performing inference..."
        prediction = sess.run(forward_model,
                              feed_dict={image_ph: np.expand_dims(img,axis=0),
                                         mask_ph: np.reshape(mask,[1,mask.shape[0],mask.shape[1],1]),
                                         edges_ph: np.expand_dims(np.expand_dims(edges,axis=0),axis=3),
                                       })
        print "Upscaling..."
        upscaled_pred = sk.transform.resize(prediction[0],[args.img_height,args.img_width, 2], preserve_range=True, order=3)

        io_utils.save_flow_file(upscaled_pred, filename='out_no_var.flo')

        print "Variational post Processing..."
        utils.calc_variational_inference_map(args.img1_filename, args.img2_filename, 'out_no_var.flo', args.out_filename, 'sintel')