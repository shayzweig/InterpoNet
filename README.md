** still a work in progress, we will upload the training code in the following weeks **  

# InterpoNet - A brain inspired neural network for optical flow dense interpolation
====================================================
This is an initial commit implementing **InterpoNet, A brain inspired neural network for optical flow dense interpolation** by Shay Zweig and Lior Wolf from Tel Aviv University [(link)](https://arxiv.org/abs/1611.09803)  

InterpoNet achieved state-of-the-art results in November 2016 on the MPI-Sintel and  KITTI2012 Optical Flow benchmarks.

The code was developed on Ubuntu 14.04, using Tensorflow. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.  

For now only the code for inference is uploaded, we will upload the training code soon.


Installation Instructions
-------------------------
1. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
2. Clone this repository by typing: `git clone https://github.com/shayzweig/InterpoNet`
3. Install all the python packages described in Requirements.txt by typing: `pip install -r Requirements.txt`
4. Make sure to configure tensorflow to your needs (GPU usage preferred)
5. Install the variational inference :
  a. type: `cd SrcVariational`
  b. type: `make` - the comilation should run without errors.

Test you installation by running the following command:
`python InterpoNet.py example/frame_0001.png example/frame_0002.png example/frame_0001.dat example/frame_0001.txt example/frame_0001.flo --ba_matches_filename=example/frame_0001_BA.txt --sintel`  
No errors should be displayed

The InterpoNet Input
-----------------------
Given two images, you need first to compute edges and matches before running InterpoNet.
You can download the code from these links.
- for edges using SED:  http://research.microsoft.com/en-us/downloads/389109f6-b4e8-404c-84bf-239f7cbf4e3d/
- that will require also Piotr Dollar's toolbox:  http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

To compute the matches - you can use any matching algorithm. The best results we obtained were on [FlowFields] (https://www.dfki.de/web/research/publications?pubid=7987)

The InterpoNet Pipeline
-----------------------

The InterpoNet pipeline consists of the following steps:

1. **Input**: example file of all the inputs is attached in the folder "example" 
  a. two images, with the same shape.
  b. A matching file produced by a matching algorithm. 
  c. Edges file produced by SED. 
2. Downsample the inputs.   
3. Calculate the bidirectional mean - if matching map from B to A was supplied.
4. Predict the dense flow map using the trained model.
5. Upsampe back to the original size.
6. Post process using variational energy minimiation.
7. **Outputs**: dense A->B optical flow field

Usage
-----
To run the Interponet pipeline, use the following syntax:  
  `python interponet.py <img1_filename> <img2_filename> <edges_filename> <matches_filename> <out_filename> [optional --model_filename model_filename] [optional --ba_matches_filename ba_matches_filename] [optional --img_width img_width] [optional --img_height img_height] [optional --downscale downscale] [optional --sintel]`  

example use:
`python InterpoNet.py example/frame_0001.png example/frame_0002.png example/frame_0001.dat example/frame_0001.txt example/frame_0001.flo --ba_matches_filename=example/frame_0001_BA.txt --sintel`  

Command Line Arguments
-----------------------
Mandatory:

img1_filename : The first image in the image pair - used in the variational post processing
img2_filename : The second image in the image pair - used in the variational post processing
edges_filename : The edges extracted by the SED edge detector
matches_filename : The output of the matching algorithm given in the format:
out_filename : The output filename

optional: 
model_filename [default='models/ff_sintel.ckpt'] : The filename of the model to use.
ba_matches_filename [default=None] : Matching file from second image in the pair to the first (B->A)
img_width [default=1024] : The width of the flow map.
img_height [default=436] : The height of the flow map.
downscale [default=8] : How much o downscale the image before running the model - 8 is the recommended value.
sintel : Use default parameters for the sintel dataset: model_name='models/ff_sintel.ckpt', img_width=1024, img_height=436, downscale=8.


Supported models:
-----------------
* ff_sintel - Pretrained on flowfields flying chairs and finetuned on flowfields sintel.
* df_fitti2012 - Pretrained on flowfields flying chairs and finetuned on discrete flow KITTI2012.

Credits
-------
The InterpoNet pipeline couldn't be achieved without the following great software pieces:
* [TensorFlow]()  

