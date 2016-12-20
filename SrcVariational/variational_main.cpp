#include <stdlib.h>
#include <string.h>
#include "image.h"
#include "io.h"
#include "variational.h"


/* show usage information */
void usage(){
    printf("usage:\n");
    printf("    ./variational_main image1 image2 outputfile [options]\n");
    printf("Performs variational inference to fine tuning a given flow. Using the same method used by EpicFlow given two images and a .flo file and store it into a .flo file\n");
    printf("Images must be in PPM, JPG or PNG format.\n");
    printf("\n");
    printf("options:\n"); 
    printf("    -h, -help                                                print this message\n");
    printf("  energy minimization parameters\n");
    printf("    -i, -iter               <int>(5)                         number of iterations for the energy minimization\n");
    printf("    -a, -alpha              <float>(1.0)                     weight of smoothness term\n");
    printf("    -g, -gamma              <float>(3.0)                     weight of gradient constancy assumption\n");
    printf("    -d, -delta              <float>(2.0)                     weight of color constancy assumption\n");
    printf("    -s, -sigma              <float>(0.8)                     standard deviation of Gaussian presmoothing kernel\n");
    printf("  predefined parameters\n");
    printf("    -sintel                                                  set the parameters to the one optimized on (a subset of) the MPI-Sintel dataset\n");
    printf("    -middlebury                                              set the parameters to the one optimized on the Middlebury dataset\n");
    printf("    -kitti                                                   set the parameters to the one optimized on the KITTI dataset\n");
    printf("\n");
}


int main(int argc, char **argv){
    if( argc<6){
        if(argc>1) fprintf(stderr,"Error, not enough arguments\n");
        usage();
        exit(1);
    }

    // read arguments
    color_image_t *im1 = color_image_load(argv[1]);
    color_image_t *im2 = color_image_load(argv[2]);
    const char *flofile = argv[3];
    const char *outputfile = argv[4];

    // prepare variables
    variational_params_t flow_params;
    variational_params_default(&flow_params);
    
    // read optional arguments 
    #define isarg(key)  !strcmp(a,key)
    int current_arg = 6;
    while(current_arg < argc ){
        const char* a = argv[current_arg++];
        if( isarg("-h") || isarg("-help") ) 
            usage();
        else if( isarg("-i") || isarg("-iter") ) 
            flow_params.niter_outer = atoi(argv[current_arg++]); 
        else if( isarg("-a") || isarg("-alpha") ) 
            flow_params.alpha= atof(argv[current_arg++]);  
        else if( isarg("-g") || isarg("-gamma") ) 
            flow_params.gamma= atof(argv[current_arg++]);                                  
        else if( isarg("-d") || isarg("-delta") ) 
            flow_params.delta= atof(argv[current_arg++]);  
        else if( isarg("-s") || isarg("-sigma") ) 
            flow_params.sigma= atof(argv[current_arg++]); 
        else if( isarg("-sintel") ){ 
            flow_params.niter_outer = 5;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;            
        }  
        else if( isarg("-kitti") ){ 
            flow_params.niter_outer = 2;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.77f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.7f; 
        }
        else if( isarg("-middlebury") ){ 
            flow_params.niter_outer = 25;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;  
        }
        else{
            fprintf(stderr, "unknown argument %s", a);
            usage();
            exit(1);
        }   
    }
    
    image_t **flow = readFlowFile(flofile);
    
    image_t *wx = flow[0];
    image_t *wy = flow[1];
    
    // energy minimization
    variational(wx, wy, im1, im2, &flow_params);
    
    // write output file and free memory
    writeFlowFile(outputfile, wx, wy);
    
    color_image_delete(im1);
    color_image_delete(im2);
    image_delete(wx);
    image_delete(wy);

    return 0;
}
