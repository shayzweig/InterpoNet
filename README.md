# InterpoNet - A brain inspired neural network for optical flow dense interpolation
====================================================
This is an initial commit implementing **InterpoNet, A brain inspired neural network for optical flow dense interpolation** by Shay Zweig and Lior Wolf from Tel Aviv University [(link)](https://arxiv.org/abs/1611.09803)  

InterpoNet achieved state-of-the-art results in November 2016 on the MPI-Sintel and  KITTI2012 Optical Flow benchmarks.

The code was developed on Ubuntu 14.04, using Tensorflow. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.  

For now only the code for inference is uploaded, we will upload the training code soon.


Installation Instructions
-------------------------
1. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
2. activate the virtualenv, by typing `source env/bin/activate`
2. Clone this repository by typing: `git clone https://github.com/shayzweig/InterpoNet`
3. Install all following python packages (using `pip install`) : numpy (tested version: 1.11.2), scikit-image (tested version: 0.12.3). (other dependencies such as cython might be required)
4. install tensorflow according to the instructions in their [website](https://www.tensorflow.org/versions/r0.10/get_started/os_setup)  . We only tested our program using tensorflow 0.10.0, and 0.11.0, it should work for versions >0.10.0 
4. Make sure to configure tensorflow to your needs (GPU usage preferred, you will need specific versions of cuda and cudnn for that)
5. Install the variational inference program:
  1. Libraries libpng, libm and liblapack are required.
  2. from the root folder of Interponet, type: `cd SrcVariational`
  3. type: `make` - the comilation should run without errors.

Test your installation by running the following command from the root folder of InterpoNet:
`python InterpoNet.py example/frame_0001.png example/frame_0002.png example/frame_0001.dat example/frame_0001.txt example/frame_0001.flo --ba_matches_filename=example/frame_0001_BA.txt --sintel`  
No errors should be displayed

The program was only tested under a 64-bit Linux distribution with tensorflow version 0.10 .
We do not give any support for compilation issues or other OS.
 
The InterpoNet Input
-----------------------
The input to the algorithm is divided into three components:
1. The two images in the image pair
they must have the same shape and they are only used for the variational energy minimization.

2. A matching file produced by a matching algorithm. 
The format of the matching file should be as follows: 
A text file, each match should be preseted in a different row which should include the source and target coordinates seperated with spaces:
x1 y1 x2 y2 

Any additional information in the row (such as match score) is discarded.

To compute the matches - you can use any matching algorithm. The best results we obtained were on [FlowFields] (https://www.dfki.de/web/research/publications?pubid=7987)

3. Edges file produced by SED.
The edges should be calculated using the SED algorithm. You can download the code from these links.
- http://research.microsoft.com/en-us/downloads/389109f6-b4e8-404c-84bf-239f7cbf4e3d/
- that will require also Piotr Dollar's toolbox:  http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html


Usage
-----
To run the InterpoNet pipeline, use the following syntax:  
  `python interponet.py <img1_filename> <img2_filename> <edges_filename> <matches_filename> <out_filename> [optional --model_filename model_filename] [optional --ba_matches_filename ba_matches_filename] [optional --img_width img_width] [optional --img_height img_height] [optional --downscale downscale] [optional --sintel]`  

Example use:
`python InterpoNet.py example/frame_0001.png example/frame_0002.png example/frame_0001.dat example/frame_0001.txt example/frame_0001.flo --ba_matches_filename=example/frame_0001_BA.txt --sintel`  

Command Line Arguments
-----------------------
Mandatory:
* img1_filename : The first image in the image pair - used in the variational post processing
* img2_filename : The second image in the image pair - used in the variational post processing
* edges_filename : The edges extracted by the SED edge detector
* matches_filename : The output of the matching algorithm given in the format:
* out_filename : The output filename

optional: 
* model_filename [default='models/ff_sintel.ckpt'] : The filename of the model to use.
* ba_matches_filename [default=None] : Matching file from second image in the pair to the first (B->A)
* img_width [default=1024] : The width of the flow map.
* img_height [default=436] : The height of the flow map.
* downscale [default=8] : How much o downscale the image before running the model - 8 is the recommended value.
* sintel : Use default parameters for the sintel dataset: model_name='models/ff_sintel.ckpt', img_width=1024, img_height=436, downscale=8.

The InterpoNet Output
-----------------------
The output flow file is in the same format as the flow maps in the MPI-Sintel dataset. The provided function: load_flow_file in io_utils.py loads the file into an h x w x 2 matrix.

Supported models:
-----------------
* ff_sintel - Pretrained on flowfields flying chairs and finetuned on flowfields sintel.
* df_fitti2012 - Pretrained on flowfields flying chairs and finetuned on discrete flow KITTI2012.

Credits
-------
InterpoNet was built with the help of the following great software pieces:
* [TensorFlow](https://www.tensorflow.org/)
* [EpicFlow] (https://thoth.inrialpes.fr/src/epicflow/)

