from scipy.ndimage import label, generate_binary_structure
from scipy import ndimage
import torch
import torchvision
from torchvision.models import *
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import jaccard_similarity_score, recall_score, f1_score
import cv2
from cv2.rgbd import registerDepth
import time
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation, binary_fill_holes, binary_opening
import skimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max

impath = './images/color_image.jpg'
bonuspath = './images/color_image_bonus.jpg'
maskpath = './images/depth_image_mask.png'
imdepth = './images/depth_image.png'
imdepthc = './images/depth_image_colored.jpg'


def select_blob(seg):
    """

    Args:
        seg (Array): binary segmentation 

    Returns:
        Array: largest blob from segmentation
    """
    labeled_array, num_features = label(seg)
    max_blop_size = 0
    index_max = 0

    for feature in range(1, num_features + 1):
        tmp_vol = np.sum(labeled_array == feature)
        if tmp_vol >= max_blop_size:
            max_blop_size = tmp_vol
            index_max = feature

    res = 1 * (labeled_array == index_max)

    return res


def count_parameters(model):
    """
    Count model parameters 
    Args:
        model (TYPE): Model

    Returns:
        int: number of parameters 
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def segment_color(net, path, gtpath, resize_val=220, dev='cpu', show_im=True, blob_selection=True, save=''):
    """
    Segment a body of a person from as single rgb image from path 

    Args:
        net (Model): Model
        path (String): Path to image 
        gtpath (String): Path to ground truth
        resize_val (int, optional): Output size of image to be resized 
        dev (str, optional): Select device for computationss
        show_im (bool, optional): If True show all plots 
        blob_selection (bool, optional): If true select largest blob in segmentation

    Returns:
        dict: Metrics 
        float: Computation time 
    """
    # Count model parameters
    # count = count_parameters(model)
    # print (count)

    img = Image.open(path)
    mask = Image.open(gtpath)

    metrics = {}

    start_time = time.time()

    # Image preprocessing
    orig_size = img.size
    trf = T.Compose([T.Resize(resize_val),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)
    img = inp[0, 0, ...]

    # Ground truth preprocessing
    inpmask = np.array(mask)[..., 0]
    inpmask = 1 * (inpmask == 255)

    # Inference
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    # Select body label
    seg = 1 * (om == 15)

    # Select blob
    if blob_selection:
        seg = select_blob(seg)

    # Resize to original size
    seg = cv2.resize(seg.astype('uint8'), orig_size,
                     interpolation=cv2.INTER_AREA)

    computation_time = time.time() - start_time

    # Uncomment to savefig
    plt.imshow(inpmask)
    plt.axis('off')
    if show_im:
        plt.show()
    # plt.savefig('./predictions/pred'+str(resize_val)+str(save)+str(blob_selection)+'.jpg')

    metrics = metric(seg, inpmask)
    print('color metrics', metrics)
    return metrics, computation_time, seg


def merge_depth( path, gtpath, color_seg, resize_val=220, dev='cpu', show_im=True, blob_selection=True, save=''):
    """
    Segment a body of a person from depth image and merge with segmentation 
    Args:
        net (Model): Model
        path (String): Path to image 
        gtpath (String): Path to ground truth
        resize_val (int, optional): Output size of image to be resized 
        dev (str, optional): Select device for computationss
        show_im (bool, optional): If True show all plots 
        blob_selection (bool, optional): If true select largest blob in segmentation

    Returns:
        dict: Metrics 
        float: Computation time 
    """
    # Count model parameters
    # count = count_parameters(model)
    # print (count)

    img = Image.open(path)
    mask = Image.open(gtpath)

    metrics = {}

    start_time = time.time()

    # Ground truth preprocessing
    inpmask = np.array(mask)[..., 0]
    inpmask = 1 * (inpmask == 255)


    ## Camera calibration
    k_rgb = np.load(
        './calibration/K_color_intrinsic.npy').astype(np.float32)
    kdepth = np.load(
        './calibration/K_depth_intrinsic.npy').astype(np.float32)
    Rt = np.load(
        './calibration/T_color_to_depth_extrinsic.npy').astype(np.float64)
    newrt = np.zeros((4, 4)).astype(np.float32)
    newrt[:3, :] = Rt
    newrt[3, :] = (0., 0., 0., 1.)
    Rt = newrt
    depth = np.asarray(Image.open(path)).astype(np.uint16)

    depth = depth[..., 0]
    k_rgb = k_rgb * 0.001
    kdepth = kdepth * 0.001
    registered = registerDepth(kdepth,
                               k_rgb, None,
                               Rt, depth, (720, 1280), depthDilation=True)




    # Use color segmentation as a region selection and compute threshold
    bcolor_seg=binary_dilation(color_seg,iterations=10).astype(color_seg.dtype)
    img=registered*bcolor_seg
    thres_seg = thres(img)


    # Select blob

    seg_blop = select_blob(thres_seg)
    seg_blop = select_blob(seg_blop*color_seg)

    computation_time = time.time() - start_time

    plt.imshow(seg_blop)
    plt.show()





    metrics = metric(seg_blop, inpmask)
    print("metrics final segmentation",metrics, 'time',computation_time)
    return computation_time, seg, inpmask



def thres(image, size=5):
    image=np.asarray(image).astype(np.uint8)
    blur=image
    blur = cv2.GaussianBlur(image, (size, size), 0)
    _,  thres_seg= cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thres_seg=binary_fill_holes(thres_seg)
    thres_seg = binary_erosion(thres_seg,iterations=2)

    return thres_seg

def metric(seg, mask):

    metrics = dict()
    metrics['recall'] = recall_score(mask.flatten(), seg.flatten())
    metrics['jindex'] = jaccard_similarity_score(
        mask.flatten(), seg.flatten())
    metrics['f1'] = f1_score(mask.flatten(), seg.flatten())

    return metrics


if __name__ == '__main__':
    model = None

    fcn = segmentation.deeplabv3_resnet101(pretrained=True).eval()
    #fcn = segmentation.fcn_resnet101(pretrained=True).eval()

    model = fcn
    for sizeim in [224]:
        for blob_selection in [True]:


            metrics,computation_time, seg = segment_color(
                model, impath, maskpath, resize_val=sizeim, show_im=False, blob_selection=blob_selection, save='resnet')

            print(metrics, "  fcn_resnet101 with blob selection " + str(blob_selection) +
                  "  ... size of image " + str(sizeim) + "--- %s seconds ---" % (computation_time))

    computation_time, merge_seg, mask = merge_depth( imdepth, maskpath, seg, resize_val=224, show_im=False, blob_selection=False, save='bonusdfcn')


