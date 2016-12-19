** still a work in progress, we will upload the training code in the following weeks **  

# InterpoNet - A brain inspired neural network for optical flow dense interpolation
====================================================
This is an initial commit implementing **InterpoNet, A brain inspired neural network for optical flow dense interpolation** by Shay Zweig and Lior Wolf from Tel Aviv University [(link)](https://arxiv.org/abs/1611.09803)  

InterpoNet achieved state-of-the-art results in November 2016 on the MPI-Sintel and  KITTI2012 Optical Flow benchmarks.

The code was developed on Ubuntu 14.04, using Tensorflow. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.  

For now only the code for inference is uploaded, we will upload the training code soon.

Installation Instructions
-------------------------
1. Download and compile OpenCV 2.4.10, with python2.7 support
2. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
3. Copy the cv2.so file which was generated in step 1 into `env/lib/python2.7/site-packages`
4. Clone this repository by typing: `git clone https://github.com/shayzweig/InterpoNet`
5. Install all the python packages described in Requirements.txt by typing: `pip install -r Requirements.txt`
6. Make sure to configure tensorflow to your needs (GPU usage preferred)

Given two images, you need first to compute edges and matches before running EpicFlow.
You can download the code from these links.
- for edges using SED:  http://research.microsoft.com/en-us/downloads/389109f6-b4e8-404c-84bf-239f7cbf4e3d/
- that will require also Piotr Dollar's toolbox:  http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

The InterpoNet Pipeline
-----------------------
The InterpoNet pipeline consists of the following steps:

1. **Input**: example file of all the inputs is attached in the folder "example" 
  1.1 two images, with the same shape.
  1.2 A matching file produced by a matching algorithm. 
  1.3 Edges file produced by SED.
2. Calculate descriptors (per each pixel in both images) using the PatchBatch CNN, i.e calculate a [h,w,#dim] tensor per
   image  
3. Find correspondences between both descriptor tensors using PatchMatch, with an L2 cost function  
4. Eliminate incorrect assignments using a bidirectional consistency check  
5. **(Not yet implemented in this repository)** Use the L2 costs + EpicFlow algorithm to interpolate the sparse optical
   flow field into a dense one (we used the default parameters of EpicFlow)
6. **Outputs**: A->B optical flow field, (optional) descriptors file, cost assignment file

Usage
-----
To run the PatchBatch pipeline, use the following syntax:  
`python patchbatch.py <img1_filename> <img2_filename> <model_name> <output_path> [optional -bidi] [optional --descs]`  

Currently supported models:
* KITTI2012_CENTSD_ACCURATE
* KITTI2015_CENTSD_ACCURATE
* KITTI2012_SPCI
* KITTI2015_SPCI

If the output_path does not exist, it will be created. In it will be placed the following:  
* flow.pickle - 
  * A [h,w,3] numpy array with channel 0,1,2 being U, V, valid flag components of the flow field 
  * If the `-bidi` flag is invoked, the code will compute 2 flow fields: img1->img2 and img2->img1 and will mark as 'invalid' all correspondences with inconsistent matchings (i.e. >1 pixels apart)
* cost.pickle - 
  * A [h,w] numpy array containing the matching cost per match
* (If the --descs option was used) descs.pickle - 
  * A list with two [h,w,#d] numpy arrays, the first contains descriptors per each pixel of img1, and the second the same for img2

You can also use `benchmark_kitti.py` to run a full benchmark on a folder with KITTI images.

For now, the EpicFlow extension is not yet implemented - so what you're getting is a pure PatchBatch descriptors + PatchMatch result.

Credits
-------
The PatchBatch pipeline couldn't be achieved without the following great software pieces:
* [Theano](https://github.com/Theano/Theano)  
* [Lasagne](https://github.com/Lasagne/Lasagne)  

We also used the following toolkit for visualization:
* [OpticalFlowToolkit](https://github.com/liruoteng/OpticalFlowToolkit)

