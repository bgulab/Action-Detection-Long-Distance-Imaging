



from tensorflow.keras import backend as K



import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import datetime
import xml.etree.ElementTree as ET
    
import glob
import cv2
from operator import itemgetter


from yolo import YOLO
from PIL import Image
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def detect_img(img,yolo):
        #img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
        yolo.close_session()

FLAGS = None

load_prev_model=1
OF='Tvl1'#'Tvl1','liteFlow'
update_label = True
only_rbg = False
only_flow = False
batch_size = 1 # Change the batch size if you like, or if you run into memory issues with your GPU.
epochs = 350

img_channels = 3 # Number of color channels of the input images
mean_color = None# [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = False#True # [2, 1, 0]The color channel order in the original SSD is BGR
n_classes = 4 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO

score_threshold=0.3
iou_threshold=0.45

max_boxes=20
mv_avrg_hist=60
grace_windows=20
frame_mAP_th=0.2
turb_life_time=15

dirnames_rgb='./data/test/RGB'



if OF == 'liteFlow':
    dirnames_flow='./data/test/LiteFlow'

elif OF == 'Tvl1':
    dirnames_flow = './data/test/Tvl1'

K.clear_session() # Clear previous models from memory.


resultsPath = './results/yolo_rbg_2d_flow_2d_'+OF
resultsPath_merge ='./results/merge_yolo_rbg_2d_flow_2d_'+OF


# TODO: Set the path to the VGG-16 weights.
model_rgb=YOLO(model_path='./weights/yolo/trained_weights_final.h5',classes_path='./data/ACT_classes.txt',anchors_path='./weights/yolo/yolo_anchors.txt')


model_flow=None




if OF == 'liteFlow':
    model_flow = YOLO(model_path='./weights/yolo/trained_weights_final_liteflow.h5',classes_path='./data/ACT_classes.txt',anchors_path='./weights/yolo/yolo_anchors.txt')


elif OF == 'Tvl1':
    model_flow = YOLO(model_path='./weights/yolo/trained_weights_final_Tvl1.h5',
                      classes_path='./data/ACT_classes.txt',anchors_path='./weights/yolo/yolo_anchors.txt')




def merge_score(y_merge,y_flow,wight):
    
    y_merge=y_merge*wight+y_flow*(1.0-wight)
    return y_merge

def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,-2]-b[:,-4]+1) * (b[:,-1]-b[:,-3]+1)
def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:,-4], b2[:,-4])
    ymin = np.maximum(b1[:,-3], b2[:,-3])
    xmax = np.minimum(b1[:,-2] + 1, b2[:,-2] + 1)
    ymax = np.minimum(b1[:,-1] + 1, b2[:,-1] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def nms2d(boxes, overlap=0.1):
    """Compute the nms given a set of scored boxes,
    as numpy array with 6 columns <class> <score> <x1> <y1> <x2> <y2> 
    return the indices of the tubelets to keep
    """

    if boxes.size == 0:
        return np.array([],dtype=np.int32)

    x1 = boxes[:, -4]
    y1 = boxes[:, -3]
    x2 = boxes[:, -2]
    y2 = boxes[:, -1]

    scores = boxes[:, -5]
    areas = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(scores)
    indices = np.zeros(scores.shape, dtype=np.int32)

    counter = 0
    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1

        xx1 = np.maximum(x1[i],x1[I[:-1]])
        yy1 = np.maximum(y1[i],y1[I[:-1]])
        xx2 = np.minimum(x2[i],x2[I[:-1]])
        yy2 = np.minimum(y2[i],y2[I[:-1]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[I[:-1]] - inter)
        I = I[np.where(iou <= overlap)[0]]

    return boxes[ indices[:counter],:]

def most_common(lst):
    return max(set(lst), key=lst.count)

def updatelabel(trk,mv_avrg_hist):
    if (len(trk["tube"])>mv_avrg_hist):
        if (trk["label"]==trk["tube"][-1][0]):
           trk["num_diff_class_hit"]=0
          
        else:
             if (trk["num_diff_class_hit"]>mv_avrg_hist):
                 trk["label"]= most_common([trk["tube"][idx][0] for idx in np.arange(-mv_avrg_hist,0) ])
                 trk["num_diff_class_hit"]=0
             else:
                 trk["num_diff_class_hit"]=trk["num_diff_class_hit"]+1
        trk_label_hist=[trk["tube"][idx][0] for idx in np.arange(-mv_avrg_hist,0) ]
        idx=np.arange(-mv_avrg_hist,0)
        indexes = [index for index in range(len(trk_label_hist)) if trk_label_hist[index] == trk["label"]]
        indexes=idx[indexes]
        trk_list_score_label= [trk["tube"][idx][1] for idx in indexes  ]
        if len(trk_list_score_label)>0:
            trk["score"]=sum(trk_list_score_label)/len(trk_list_score_label)
        else:
            trk["score"] =0.0
           
    else:
       trk["label"]= most_common([trk_tmp[0] for trk_tmp in trk["tube"]])
       trk_label_hist=[trk_tmp[0] for trk_tmp in trk["tube"]]
       indexes = [index for index in range(len(trk_label_hist)) if trk_label_hist[index] == trk["label"]]
       trk_list_score_label= [trk["tube"][idx][1] for idx in indexes  ]
       if len(trk_list_score_label) > 0:
           trk["score"] = sum(trk_list_score_label) / len(trk_list_score_label)
       else:
           trk["score"] = 0.0
            
    return trk["score"],trk["label"]

def clacVel(trk,mv_avrg_hist):
    last_box=trk["tube"][-1]
    Cm_x =  last_box[-4]+(last_box[-2] - last_box[-4]) / 2
    Cm_y =  last_box[-3]+ (last_box[-1] - last_box[-3]) / 2

    if (trk["num_hit"] >= mv_avrg_hist):

        last_1_box = trk["tube"][-mv_avrg_hist+1]

        Cm_x_last= last_1_box[-4]+ (last_1_box[-2]-last_1_box[-4] )/2
        Cm_y_last= last_1_box[-3] +  (last_1_box[-1]-last_1_box[-3] )/2

        VelX_last = (Cm_x-Cm_x_last)/mv_avrg_hist
        VelY_last = (Cm_y-Cm_y_last)/mv_avrg_hist

        
        trk["VelX"]=(trk["VelX"]*(mv_avrg_hist-1)+VelX_last)/mv_avrg_hist
        trk["VelY"]=(trk["VelY"]*(mv_avrg_hist-1)+VelY_last)/mv_avrg_hist
    else:

        last_1_box = trk["tube"][0]

        Cm_x_last =last_1_box[-4]+ (last_1_box[-2] - last_1_box[-4]) / 2
        Cm_y_last = last_1_box[-3] +(last_1_box[-1] - last_1_box[-3]) / 2

        VelX_last = (Cm_x - Cm_x_last)/trk["num_hit"]
        VelY_last = (Cm_y - Cm_y_last)/trk["num_hit"]

        trk["VelX"]=VelX_last
        trk["VelY"]=VelY_last
    
    return trk["VelX"], trk["VelY"] 
def update_trk(trk):
    box=trk["tube"][-1]
    box[-4]= trk["VelX"] + box[-4]
    box[-3]= trk["VelY"] + box[-3]
    box[-2]= trk["VelX"] + box[-2]
    box[-1]= trk["VelY"] + box[-1]
    
    trk["tube"].append(box)
    return trk["tube"]
def pr_to_ap(pr,label_class,zoom_str):
    """Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """
    total_gt=max(pr[:, 2])
    precision=pr[1:, 0]/(pr[1:, 0]+pr[1:, 1]+np.finfo(float).eps)
    if total_gt>0:
       recall=pr[1:, 0]/total_gt
    else:
       recall=np.ones(len(pr[1:, 0]))
    prdif = recall[1:] - recall[:-1]
    prsum = precision[1:] + precision[:-1]
    plt.plot(precision, recall, 'o',label=zoom_str + ' ' +label_class)

    return np.sum(prdif * prsum*0.5 )
def loadgt(datasetpath,imgname,classes):
    
    
    
    gt = []
    imgname_tmp=imgname.split(".")
    FileString=datasetpath+os.sep+imgname_tmp[0]+'.xml'
    try:
      file_name = open(FileString, 'r')
    except IOError:
      stringtoWrite = "Error: File " + (FileString)+ " does not appear to exist."
      print (stringtoWrite)
      return  gt 
    #remove NUL charcter
    with open(FileString, 'rb+') as filehandle:
      filehandle.seek(-1, os.SEEK_END)
      charchterString = filehandle.read(1)
      #nullExist = re.search('x00',charchterString)
      if (charchterString == b'\x00' ):
          filehandle.seek(-1, os.SEEK_END)
          filehandle.truncate()



   
    tree = ET.parse(file_name)

    root = tree.getroot()
   # print("-root is ",root.tag)

   # print("\n-all root childs:")
   ##########################################PARSING DOWN####################################
    childsize=root.find("size")
    #scalereduceX = img_width / int(childsize.find("width").text)
    #scalereduceY = img_height / int(childsize.find("height").text)
    scalereduceX=1
    scalereduceY=1
    for child in root.findall("object"):
           label = child.find("name").text
           label_idx=classes.index(label) 
                     
           Xmin = int(child.find("bndbox")[0].text)
           Ymin = child.find("bndbox")[1].text
           Xmax = child.find("bndbox")[2].text
           Ymax = child.find("bndbox")[3].text
            
           gt.append([label_idx,int(int(Xmin)*scalereduceX),int(int(Ymin)*scalereduceY),int(int(Xmax)*scalereduceX),int(int(Ymax)*scalereduceY)])
    return gt

def checkClass(datasetpath,imgname,detection,label_list,classTrk_list):
      gt_list=loadgt(datasetpath,imgname,label_list)
      total_gt_list_label=0
      for classTrk in classTrk_list:
            gt_list_label = [gt_list[index] for index in range(len(gt_list)) if gt_list[index][0] == classTrk]
            total_gt_list_label=total_gt_list_label+len(gt_list_label)
    
      if total_gt_list_label>0:
          return True
      else:
          return False


def decode_detections(boxes,box_scores,box_scores_rgb):


    mask = (box_scores >= score_threshold) & (box_scores_rgb >= score_threshold)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(n_classes):
        # TODO: use keras backend instead of tf.
        boxes_tmp=[]
        if any(mask[:, c]):
            boxes_tmp =  boxes[ mask[:, c],:]
            class_boxes = boxes_tmp[:, np.array([1, 0, 3, 2])]
            class_box_scores = box_scores[mask[:, c], c]
            class_boxes = nms2d(np.concatenate((np.expand_dims(class_box_scores, axis=-1), class_boxes), axis=-1),iou_threshold)

            classes = np.ones(class_boxes.shape[0]) * (c+1)
            if not (class_boxes.size == 0):
                if len(boxes_)==0:
                    boxes_=np.concatenate((np.expand_dims(classes, axis=-1), class_boxes), axis=-1)
                else:
                    boxes_=np.concatenate((boxes_,np.concatenate((np.expand_dims(classes, axis=-1), class_boxes), axis=-1)), axis=0)
    return np.array(boxes_)

def frameAP(res,datasetpath,imgname,detection,label_list,zoom_str, th=0.5,debugflag = False,res_detect_only=[]):
            label_str_tmp =  'human_'+imgname.split('.')[0].split('_')[1]
            for label_str in label_list:
              if  label_str_tmp in  label_str :
                  break
            # load ground-truth of this class
            gt_list=loadgt(datasetpath,imgname,label_list)
            if len(res_detect_only)>0:
                gt_list_detect_only=copy.deepcopy(gt_list)

                gt_total_detect_only = len(gt_list_detect_only)  # false negatives
                fp_detect_only = 0  # false positives
                tp_detect_only = 0  # true positives
                for detect in detection:
                    box_tmp = np.array(detect[-4:])
                    ispositive = False
                    if len(gt_list_detect_only) > 0:
                        ious = iou2d(np.asarray(gt_list_detect_only), box_tmp)
                        amax = np.argmax(ious)
                        if ious[amax] >= th:
                            ispositive = True
                            gt_list_detect_only = np.delete(gt_list_detect_only, (amax), axis=0)
                    if ispositive:
                        tp_detect_only += 1
                    else:
                        fp_detect_only += 1
            # pr will be an array containing precision-recall values
                if (len(detection) > 0 or len(gt_list) > 0):
                        res_detect_only[zoom_str][label_str].append(
                            [tp_detect_only + res_detect_only[zoom_str][label_str][-1][0], fp_detect_only + res_detect_only[zoom_str][label_str][-1][1],
                             gt_total_detect_only + res[zoom_str][label_str][-1][2]])
            idx=1
            for label in label_list:
                if label=='background':
                    continue
                
                gt_list_label = [gt_list[index] for index in range(len(gt_list)) if gt_list[index][0] == idx]
 
                                       
                gt_total = len(gt_list_label)# false negatives
                fp = 0 # false positives
                tp = 0 # true positives                  
            
                detection_label=[detection[index] for index in range(len(detection)) if detection[index][0] == idx]
                
                for detect in detection_label:
                        
                            box_tmp = np.array(detect[-4:])
                            ispositive = False
                                        
                            if   len(gt_list_label)  >0:       
                                ious = iou2d(np.asarray(gt_list_label), box_tmp)
                                amax = np.argmax(ious)
                                                
                                if ious[amax] >= th:
                                      ispositive = True
                                      gt_list_label = np.delete(gt_list_label, (amax), axis=0)


                            if ispositive:
                                            tp += 1
                            else:
                                if debugflag:
                                    if len(gt_list_label) > 0:
                                        print("false : " + datasetpath + ' ' + imgname + ' Score :' + str(detect[1]) + ' iou2d: ' +str(ious[amax]) )
                                    else:
                                        print("false : " + datasetpath + ' ' + imgname + ' Score :' + str(detect[1]) + ' iou2d: ' + str(0))
                                fp += 1
                idx=idx+1
                if   (len(detection_label)>0 or  len(gt_list_label)>0):

                    res[zoom_str][label].append([tp +  res[zoom_str][label][-1][0],fp +  res[zoom_str][label][-1][1],gt_total +  res[zoom_str][label][-1][2]])
            if len(res_detect_only) > 0:
                 return res,res_detect_only
            else:
                return res
    
  

def int_trk(trk_tmp,trk_id):
    trk_tmp = {}
    trk_tmp["id"]=trk_id+1
    trk_tmp["num_hit"] = 0
    trk_tmp["num_miss"] = 0
    trk_tmp["num_diff_class_hit"] = 0
    trk_tmp["score"] = 0.0
    trk_tmp["label"] = 0
    trk_tmp["tube"] = []
    trk_tmp["VelX"] = 0.0
    trk_tmp["VelY"] = 0.0

    return trk_tmp



colors_tmp = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
colors = np.asarray(colors_tmp)[:,:3] * 255
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
classes = ['background', 'human_walking', 'human_crawl', 'human_running', 'human_walkingCrouched']
clip_name = ['walking_3','running_5','running_4','walkingCrouche_4','walkingCrouche_3','crawl_1']
zoom_list = ['X100','X50']


res_trk={}
res_trk_detect_only={}
res_rgb={}
res_flow={}
res_rgb_mereg={}

res_trk_label={}
res_rgb_label={}
res_flow_label={}
res_rgb_mereg_label={}

trk_tmp = {}
res_tmp_label={}
res_rgb_mereg_detect_only={}
for label in classes:
    if label != 'background':
        res_tmp_label[label] = []

        res_tmp_label[label].append([0, 0, 0])

for zoom_str in zoom_list:
    res_trk_detect_only[zoom_str]=copy.deepcopy(res_tmp_label)
    res_trk[zoom_str] = copy.deepcopy(res_tmp_label)
    res_rgb[zoom_str] = copy.deepcopy(res_tmp_label)
    res_flow[zoom_str] = copy.deepcopy(res_tmp_label)
    res_rgb_mereg[zoom_str] = copy.deepcopy(res_tmp_label)
    res_rgb_mereg_detect_only[zoom_str] = copy.deepcopy(res_tmp_label)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')


now=datetime.datetime.now()
if os.path.isdir(resultsPath_merge):
    os.rename(resultsPath_merge,resultsPath_merge + '_' + now.strftime("%d%m%y%H%M%S"))
if os.path.isdir(resultsPath):
    os.rename(resultsPath, resultsPath + '_' + now.strftime("%d%m%y%H%M%S"))

os.makedirs(resultsPath_merge)
os.makedirs(resultsPath)

tempval = os.getcwd()
for zoom_str in zoom_list:
    for clip_name_i in clip_name:
        trk_id = 0
        trk_tmp = int_trk(trk_tmp,trk_id)
        list_trk=[]
        os.chdir(dirnames_rgb)

        DirList_jpg = glob.glob(zoom_str+'_'+clip_name_i+"*.jpg")
        print(zoom_str+'_'+clip_name_i+"*.jpg")
        frame_num=1
        os.chdir(tempval)
        if len(DirList_jpg)>0:
            vid = cv2.VideoWriter()
            vid_detect_only = cv2.VideoWriter()
            vid.open(resultsPath + os.sep + zoom_str + '_' + clip_name_i + '_with_trk.avi', fourcc, 25.0,(854,480) )
            vid_detect_only.open(resultsPath + os.sep + zoom_str + '_' + clip_name_i + '.avi', fourcc, 25.0,(854,480) )

            for imgname in DirList_jpg:

                y_pred_decoded_merge_05=[]
                y_pred_decoded_rgb=[]
                y_pred_decoded_flow=[]
                y_pred_merge_05=[]
                y_flow=[]
                y_rgb=[]
                X_rgb=[]
                X_flow=[]
                y_merge_05=[]
                boxes_merge_05=[]
                boxes_rgb=[]
                boxes_flow=[]
                Rgb_boxes_raw=[]
                flow_boxes_raw=[]
                Rgb_box_scores_raw=[]
                flow_box_scores_raw=[]
                image_rgb=[]
                image_flow=[]


                filenames_rgb=dirnames_rgb+ os.sep+imgname
                filenames_flow=dirnames_flow+ os.sep+imgname

                image_rgb,Rgb_box_scores_raw ,Rgb_boxes_raw=model_rgb.detect_image_without_encoding(filenames_rgb)
                image_flow,flow_box_scores_raw ,flow_boxes_raw=model_flow.detect_image_without_encoding(filenames_flow)




                wight=0.5
                y_merge_05 = np.copy(Rgb_box_scores_raw)
                y_pred_merge_05=merge_score(y_merge_05,flow_box_scores_raw,wight)
                boxes_merge_05  = decode_detections(Rgb_boxes_raw,y_pred_merge_05,Rgb_box_scores_raw)
                boxes_rgb  = decode_detections(Rgb_boxes_raw,Rgb_box_scores_raw,Rgb_box_scores_raw)
                boxes_flow = decode_detections(flow_boxes_raw,flow_box_scores_raw,flow_box_scores_raw)


                i = 0 # Which batch item to look at


                if (len(boxes_rgb)>0):
                    res_rgb=frameAP(res_rgb,dirnames_rgb,imgname,nms2d( boxes_rgb),classes,zoom_str, frame_mAP_th)
                else:
                    res_rgb=frameAP(res_rgb,dirnames_rgb,imgname,[],classes,zoom_str, 0.5)

                if (len(boxes_flow)>0):
                    res_flow=frameAP(res_flow,dirnames_rgb,imgname,nms2d(boxes_flow),classes,zoom_str, frame_mAP_th)
                else:
                    res_flow=frameAP(res_flow,dirnames_rgb,imgname,[],classes, zoom_str,frame_mAP_th)

                boxs=[]
                label=[]
                img=np.copy(image_rgb)
                img_merge=np.copy(image_rgb)

                detection_mAP=[]
                if only_flow:
                    boxs = nms2d(boxes_flow)
                elif only_rbg:
                    boxs = nms2d(boxes_rgb)
                elif (len(boxes_rgb)>0 and len(boxes_flow)>0 and len(boxes_merge_05)>0 ):
                      boxs_rbg= nms2d( boxes_rgb)
                      boxs_flow= nms2d( boxes_flow)
                      score_rgb=np.max(boxs_rbg[:,1])
                      score_flow=np.max(boxs_flow[:,1])
                      if score_flow>score_rgb:
                          boxs =nms2d(boxes_merge_05,0.01)
                      else:
                          boxs =nms2d( boxes_rgb)
                elif (len(boxes_rgb)>0):
                       boxs =nms2d( boxes_rgb)
                res_rgb_mereg,res_rgb_mereg_detect_only=frameAP(res_rgb_mereg,dirnames_rgb,imgname,boxs,classes,zoom_str,frame_mAP_th,True,res_rgb_mereg_detect_only)

                label_str_tmp = 'human_' + imgname.split('.')[0].split('_')[1]

                for box in boxs:
                    xmin = box[-4]
                    ymin = box[-3]
                    xmax = box[-2]
                    ymax = box[-1]
                    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                    if label_str_tmp in classes[int(box[0])]:
                        img_merge = cv2.rectangle(img_merge, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0, 255, 0), 2)

                        img_merge = cv2.putText(img_merge, label, (int(xmin), int(ymin)), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        img_merge = cv2.rectangle(img_merge, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        img_merge = cv2.putText(img_merge, label, (int(xmin), int(ymin)), font, 0.6,  (0, 0, 255), 1,
                                                cv2.LINE_AA)
                cv2.imwrite(resultsPath_merge+os.sep+imgname,img_merge)
                vid_detect_only.write(img_merge)
                if (len(boxs)>0):

                    if (len(list_trk)>0):
                         list_trk=sorted(list_trk, key=itemgetter('num_hit'),reverse=True)

                         for trk in list_trk:
                           updated_flag=0
                           boxs_loop=boxs
                           i=0
                           for box in boxs_loop:
                                    iou2d_tmp=iou2d( trk["tube"][-1],box)

                                    if iou2d_tmp>0.2 :
                                        trk["num_hit"] = trk["num_hit"] + 1
                                        trk["num_miss"] = 0


                                        updated_flag = 1
                                        if ((iou2d_tmp<0.35) and (trk["num_hit"] > mv_avrg_hist)) :
                                            trk["tube"] = update_trk(trk)
                                            box = trk["tube"][-1]
                                        else:
                                            trk["tube"].append(box)
                                            trk["VelX"], trk["VelY"] = clacVel(trk, mv_avrg_hist)


                                        xmin = box[-4]
                                        ymin = box[-3]
                                        xmax = box[-2]
                                        ymax = box[-1]

                                        color = colors[trk["id"]%len(colors)]

                                        if update_label:
                                            trk["score"], trk["label"] = updatelabel(trk, mv_avrg_hist)
                                        else:
                                            trk["label"] = box[0]
                                            trk["score"] = box[1]
                                        detection_mAP.append([int(trk["label"]) , xmin,ymin,xmax,ymax])
                                        #label = '{},{:.2f},{}.{}: {:.2f}'.format(int(trk["id"]),float(iou2d_tmp),int(trk["label"]),classes[int(trk["label"])], trk["score"])


                                        label = '{}: {:.2f}'.format(classes[int(trk["label"])],trk["score"])

                                        if  label_str_tmp in classes[int(trk["label"])]:
                                            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)

                                            img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,  (0,255,0), 1, cv2.LINE_AA)
                                        else:
                                            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                                (0, 0, 255), 2)

                                            img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,
                                                              (0, 0, 255), 1, cv2.LINE_AA)
                                        boxs = np.delete(boxs, (i), axis=0)
                                        i=i-1
                                        break
                                    i=i+1
                           if (updated_flag==0):
                                 if (trk["num_miss"]>grace_windows or trk["num_hit"] < turb_life_time ):
                                     list_trk.remove(trk)
                                 else:
                                     trk["num_miss"] = trk["num_miss"]+1
                                     trk["tube"] = update_trk(trk)
                                     color = colors[trk["id"] % len(colors)]

                                     #label = '{},{}.{}: {:.2f}'.format(int(trk["id"]), int(trk["label"]),
                                     #                                  classes[int(trk["label"])], trk["score"])

                                     box = trk["tube"][-1]
                                     xmin = box[-4]
                                     ymin = box[-3]
                                     xmax = box[-2]
                                     ymax = box[-1]
                                     detection_mAP.append([int(trk["label"]), xmin, ymin, xmax, ymax])
                                     label = '{}: {:.2f}'.format(classes[int(trk["label"])], trk["score"])

                                     if label_str_tmp in classes[int(trk["label"])]  :
                                         img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                             (0, 255, 0), 2)

                                         img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6, (0, 255, 0),
                                                           1, cv2.LINE_AA)
                                     else:
                                         img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                             (0, 0, 255), 2)

                                         img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,
                                                           (0, 0, 255), 1, cv2.LINE_AA)



                         boxs_loop=boxs
                         box_id = 0
                         for box in boxs_loop:
                                    xmin = box[-4]
                                    ymin = box[-3]
                                    xmax = box[-2]
                                    ymax = box[-1]
                                    color = colors[int(box_id+trk_tmp["id"])%len(colors)]

                                    trk_tmp["num_hit"]=1
                                    trk_tmp["score"]= box[1]
                                    trk_tmp["label"]=int(box[0])
                                    trk_tmp["tube"] = []
                                    trk_tmp["tube"].append(box)
                                    trk_tmp["VelX"]=0.0
                                    trk_tmp["VelY"]=0.0
                                    detection_mAP.append([int(box[0]), xmin,ymin,xmax,ymax])

                                    list_trk.append(copy.deepcopy(trk_tmp))


                                    #label = '{},{}.{}: {:.2f}'.format(trk_tmp["id"],int(box[0]),classes[int(box[0])], box[1])
                                    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

                                    if label_str_tmp in classes[int(box[0])]  :
                                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                            (0, 255, 0), 2)

                                        img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6, (0, 255, 0),
                                                          1, cv2.LINE_AA)
                                    else:
                                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                            (0, 0, 255), 2)

                                        img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,
                                                          (0, 0, 255), 1, cv2.LINE_AA)


                                    box_id = box_id + 1
                                    trk_id = trk_id + 1
                                    trk_tmp = int_trk(trk_tmp, trk_id)
                         cv2.imwrite(resultsPath+os.sep+imgname,img)
                         vid.write(img)
                    else :
                        box_id=0
                        for box in boxs:
                                    xmin = box[-4]
                                    ymin = box[-3]
                                    xmax = box[-2]
                                    ymax = box[-1]
                                    color = colors[box_id%len(colors)]

                                    trk_tmp["num_hit"]=1
                                    trk_tmp["score"]= box[1]
                                    trk_tmp["label"]=int(box[0])
                                    trk_tmp["tube"] = []
                                    trk_tmp["tube"].append(box)
                                    trk_tmp["VelX"]=0.0
                                    trk_tmp["VelY"]=0.0
                                    detection_mAP.append([int(box[0]), xmin,ymin,xmax,ymax])

                                    list_trk.append(copy.deepcopy(trk_tmp))



                                    #label = '{},{}.{}: {:.2f}'.format(trk_tmp["id"],int(box[0]),classes[int(box[0])], box[1])

                                    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

                                    if label_str_tmp in classes[int(box[0])] :
                                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                            (0, 255, 0), 2)

                                        img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6, (0, 255, 0),
                                                          1, cv2.LINE_AA)
                                    else:
                                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                            (0, 0, 255), 2)

                                        img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,
                                                          (0, 0, 255), 1, cv2.LINE_AA)

                                    box_id=box_id+1
                                    trk_id = trk_id + 1
                                    trk_tmp = int_trk(trk_tmp, trk_id)
                        cv2.imwrite(resultsPath+os.sep+imgname,img)
                        vid.write(img)
                elif (len(list_trk)>0):
                    list_trk = sorted(list_trk, key=itemgetter('num_hit'), reverse=True)

                    for trk in list_trk:


                        if (trk["num_miss"] > grace_windows or trk["num_hit"] < turb_life_time):
                            list_trk.remove(trk)
                        else:
                            trk["num_miss"] = trk["num_miss"] + 1
                            trk["tube"] = update_trk(trk)
                            color = colors[trk["id"]%len(colors)]

                            #label = '{},{}.{}: {:.2f}'.format(int(trk["id"]), int(trk["label"]),
                            #                                 classes[int(trk["label"])], trk["score"])

                            box = trk["tube"][-1]
                            xmin = box[-4]
                            ymin = box[-3]
                            xmax = box[-2]
                            ymax = box[-1]

                            label = '{}: {:.2f}'.format(classes[int(trk["label"])], trk["score"])

                            if label_str_tmp in classes[int(trk["label"])] :
                                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                    (0, 255, 0), 2)

                                img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6, (0, 255, 0),
                                                  1, cv2.LINE_AA)
                            else:
                                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                    (0, 0, 255), 2)

                                img = cv2.putText(img, label, (int(xmin), int(ymin)), font, 0.6,
                                                  (0, 0, 255), 1, cv2.LINE_AA)


                            detection_mAP.append([int(trk["label"]), xmin, ymin, xmax, ymax])



                    cv2.imwrite(resultsPath + os.sep + imgname, img)
                    vid.write(img)
                else:
                    cv2.imwrite(resultsPath+os.sep+imgname,img)
                    vid.write(img)
                res_trk ,res_trk_detect_only= frameAP (res_trk,dirnames_rgb,imgname,detection_mAP,classes,zoom_str, frame_mAP_th,False,res_trk_detect_only)
                frame_num=frame_num+1
            vid.release()
            vid_detect_only.release()
ap_trk_detect_only=[]
for zoom_str in zoom_list:
    ap_trk_detect_only .append(100*np.array([pr_to_ap(np.asarray(res_trk_detect_only[zoom_str][label]),label,zoom_str) for label in classes if label !='background'])) #and res_trk[zoom_str][label][-1][2]>0'''

ap_merge_detect_only=[]
for zoom_str in zoom_list:
    ap_merge_detect_only .append(100*np.array([pr_to_ap(np.asarray(res_rgb_mereg_detect_only[zoom_str][label]),label,zoom_str) for label in classes if label !='background'])) #and res_trk[zoom_str][label][-1][2]>0'''
ap_trk=[]
ap_rgb=[]
ap_flow=[]
ap_merge=[]

for zoom_str in zoom_list:

    ap_trk .append(100*np.array([pr_to_ap(np.asarray(res_trk[zoom_str][label]),label,zoom_str) for label in classes if label !='background'])) #and res_trk[zoom_str][label][-1][2]>0'''

for zoom_str in zoom_list:

    ap_rgb.append(100*np.array([pr_to_ap(np.asarray(res_rgb[zoom_str][label]),label,zoom_str) for label in classes if label !='background' ])) #and res_rgb[zoom_str][label][-1][2]>0'''

for zoom_str in zoom_list:

    ap_flow .append(100*np.array([pr_to_ap(np.asarray(res_flow[zoom_str][label]),label,zoom_str) for label in classes if label !='background']))

for zoom_str in zoom_list:

    ap_merge .append(100*np.array([pr_to_ap(np.asarray(res_rgb_mereg[zoom_str][label]),label,zoom_str) for label in classes if label !='background' ]))

print ("frameAP trk detect only ")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0,len(classes)-1):
         print("{:20s} {:20s} {:8.2f}".format(zoom_str , classes[idx+1], ap_trk_detect_only[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str , "mAP", np.mean(ap_trk_detect_only[idx_zoom][:])))
    idx_zoom=idx_zoom+1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_trk_detect_only[index][:]) for index in np.arange(len(ap_trk_detect_only))]))))
print("")



print ("frameAP merge detect only ")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0,len(classes)-1):
         print("{:20s} {:20s} {:8.2f}".format(zoom_str , classes[idx+1], ap_merge_detect_only[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str , "mAP", np.mean(ap_merge_detect_only[idx_zoom][:])))
    idx_zoom=idx_zoom+1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_merge_detect_only[index][:]) for index in np.arange(len(ap_merge_detect_only))]))))
print("")


print ("frameAP trk")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0,len(classes)-1):
         print("{:20s} {:20s} {:8.2f}".format(zoom_str , classes[idx+1], ap_trk[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str , "mAP", np.mean(ap_trk[idx_zoom][:])))
    idx_zoom=idx_zoom+1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_trk[index][:]) for index in np.arange(len(ap_trk))]))))
print("") 

print ("frameAP merge")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0,len(classes)-1):
        print("{:20s} {:20s} {:8.2f}".format(zoom_str,classes[idx+1], ap_merge[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str, "mAP", np.mean(ap_merge[idx_zoom][:])))
    idx_zoom = idx_zoom + 1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_merge[index][:]) for index in np.arange(len(ap_merge))]))))
print("")

print("frameAP rgb")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0, len(classes) - 1):
        print("{:20s} {:20s} {:8.2f}".format(zoom_str,classes[idx + 1], ap_rgb[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str, "mAP", np.mean(ap_rgb[idx_zoom][:])))
    idx_zoom = idx_zoom + 1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_rgb[index][:]) for index in np.arange(len(ap_rgb))]))))
print("")

print("frameAP flow")
idx_zoom=0
for zoom_str in zoom_list:
    for idx in np.arange(0, len(classes) - 1):
        print("{:20s} {:20s} {:8.2f}".format(zoom_str,classes[idx + 1], ap_flow[idx_zoom][idx]))
    print("{:20s} {:20s} {:8.2f}".format(zoom_str, "mAP", np.mean(ap_flow[idx_zoom][:])))
    idx_zoom = idx_zoom + 1
print("{:20s} {:8.2f}".format("mAP", np.mean(np.array([np.mean(ap_flow[index][:]) for index in np.arange(len(ap_flow))]))))
print("")

