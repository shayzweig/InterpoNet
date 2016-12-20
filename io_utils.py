import numpy as np
import os
import struct

def save_flow_file(flow, filename):
    TAG_STRING = 'PIEH'
    # sanity check

    if filename == '':
        print ('writeFlowFile: empty filename')
        return

    idx = filename.rfind('.')

    if idx == -1:
        print ('writeFlowFile: extension required in filename %s' % filename)
        return


    if filename[idx:] != '.flo':
        print ('writeFlowFile: filename %s should have extension ''.flo''' % filename)
        return

    (height, width, nBands) = flow.shape

    if nBands != 2:
        print ('writeFlowFile: image must have two bands');
        return

    with open(filename, 'wb') as f:
        f.write(TAG_STRING)
        f.write(struct.pack('<i', width))
        f.write(struct.pack('<i', height))
        np.asarray(flow, np.float32).tofile(f)

def load_edges_file(edges_file_name, width, height):
    edges_img = np.ndarray((height,width),dtype=np.float32)
    with open(edges_file_name, 'rb') as f:
        f.readinto(edges_img)
    return edges_img

def load_matching_file(filename, width, height):
    img = np.zeros([height,width,2])
    mask = -np.ones([height,width])

    if os.path.getsize(filename) == 0:
        print ('empty file: %s' % filename)
    else:
        x1,y1,x2,y2 = np.loadtxt(filename, dtype=np.float32, delimiter=' ', unpack=True, usecols=(0,1,2,3))

        img[np.array(y1, dtype=int),np.array(x1, dtype=int),:] = np.stack((x2 - x1, y2 - y1), axis=1)
        mask[np.array(y1, dtype=int),np.array(x1, dtype=int)] = 1
        if np.any(np.isnan(img)):
            print ("Nan value found")

    return img, mask
