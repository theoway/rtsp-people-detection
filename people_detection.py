from asyncio import base_subprocess
from cProfile import label
from unittest import result
from cv2 import resize
from pip import main
import torch
import numpy as np
import cv2
from time import sleep, time

from reid import REID
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching

import threading
import queue
import operator

import warnings
warnings.filterwarnings('ignore')

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id),(x1, y1+30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,255),thickness=text_thickness)

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, devices):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devices = devices
        self.feats_queue = queue.Queue()
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.classes = [0]
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 8)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
    
    def _extract_features(self, feats, q) -> None:
        global reid, stop_threads
        print("Extracting thread has started!!!!!!!!")
        while not stop_threads:
            if not q.empty():
                id, images = q.get()
                feats[id] = reid._features(images)
                print("Person ID: ", id, '\n', "Feature length: ", feats[id].shape[0])

        print("Exiting extracting thread!!!!!!!!")

    def detection_on_stream(self, device, url):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        device = int(device)
        if device > 1:
            print("Halting 2nd device for 30 secs")
            sleep(30)

        global reid
        global threshold
        global exist_ids
        global images_by_id
        global final_fuse_id
        global feats

        frame_cnt = 0
        ids_per_frame = []
        images_by_id[device] = dict()
        
        def box_transform(box: list, shape: tuple):
            """
            shape: (frame.shape[1], frame.shape[0])
            Return box with [top_x, top_y, w, h]
            """
            x = int(box[0]*shape[0])
            y = int(box[1]*shape[1])
            w = int((box[2]-box[0]) * shape[0])
            h = int((box[3]-box[1]) * shape[1])

            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            
            #print(box, "  ------------  ", [x, y, w, h])
            return [x, y, w, h]

        cap = cv2.VideoCapture() 
        cap.open("http://{}/video".format(url))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        id_prefix = str(device) + "_"

        while True:
            track_cnt = dict()
            
            t1 = time()
            ret, frame = cap.read()
            assert ret
            print("Frame reading time: ", time() - t1)
            
            t1 = time()
            results = self.score_frame(frame)
            #frame = self.plot_boxes(results, frame)
            
            boxs = [box_transform(cords, (frame.shape[1], frame.shape[0])) for cords in results[1]] #[minx, miny, w, h]

            features = encoder(frame, boxs) # n * 128
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
            text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
            print("YOLO detection time: ", time() - t1)

            t1 = time()
            tracker.predict()
            tracker.update(detections)
            tmp_ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                    
                bbox = track.to_tlbr()
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                ids = str(id_prefix + str(track.track_id))

                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                    tmp_ids.append(ids)
                    if ids not in images_by_id[device]:
                        track_cnt[ids] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                        images_by_id[device][ids] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                    else:
                        track_cnt[ids] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                        images_by_id[device][ids].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            ids_per_frame.append(set(tmp_ids))
            print("IDs per frame: ", ids_per_frame)
            print("Object tracking time: ", time() - t1)

            for i in images_by_id[device]:
                self.feats_queue.put((i, images_by_id[device][i]))
            '''for i in images_by_id[device]:
                #print('ID number {} -> Number of images {}'.format(i, len(images_by_id[device][i])))
                t1 = time()
                feats[i] = reid._features(images_by_id[device][i])
                print("---------------------")
                print(feats[i])
                print(feats[i].shape[0])
                print("Feature generation time for ID ", i ," : ", time() - t1)'''

            t1 = time()
            for f in ids_per_frame:
                if f:
                    if len(exist_ids) == 0:
                        for i in f:
                            final_fuse_id[i] = [i]
                        exist_ids = exist_ids or f
                    else:
                        print("Exist IDs: ", exist_ids, "--------!!!!!!!!!!!!!!!!!!")
                        new_ids = f - exist_ids
                        for nid in new_ids:
                            dis = []
                            if not nid in feats.keys() or feats[nid].shape[0] < 10:
                                exist_ids.add(nid)
                                continue
                            unpickable = []
                            for i in f:
                                for key,item in final_fuse_id.items():
                                    if i in item:
                                        unpickable += final_fuse_id[key]
                            print('----------------- exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                            for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                                tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                                print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                dis.append([oid, tmp])
                            exist_ids.add(nid)
                            if not dis:
                                final_fuse_id[nid] = [nid]
                                continue
                            dis.sort(key=operator.itemgetter(1))
                            if dis[0][1] < threshold:
                                combined_id = dis[0][0]
                                images_by_id[int(combined_id[0:combined_id.find('_'):])][combined_id] += images_by_id[int(nid[0:nid.find('_'):])][nid]
                                final_fuse_id[combined_id].append(nid)
                            else:
                                final_fuse_id[nid] = [nid]

                        unpickable = []
                        for i in f:
                            for key,item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        for left_out_id in f & (exist_ids - set(unpickable)):
                            dis = []
                            if not left_out_id in feats.keys() or feats[left_out_id].shape[0] < 10:
                                continue
                            for main_id in final_fuse_id.keys():
                                tmp = np.mean(reid.compute_distance(feats[left_out_id],feats[main_id]))
                                print('------------------- Left out {}, Main ID {}, tmp {}'.format(left_out_id, main_id, tmp))
                                dis.append([main_id, tmp])
                            if dis:
                                dis.sort(key=operator.itemgetter(1))
                                print("Closest match found b/w: ", dis[0][0], left_out_id, dis[0][1])
                                if dis[0][1] < threshold:
                                    print("Creating subIDs: ", dis[0][0], left_out_id, dis[0][1])
                                    combined_id = dis[0][0]
                                    images_by_id[int(combined_id[0:combined_id.find('_'):])][combined_id] += images_by_id[int(left_out_id[0:left_out_id.find('_'):])][left_out_id]
                                    final_fuse_id[combined_id].append(left_out_id)
                                else:
                                    print("New ID added: ", left_out_id)
                                    final_fuse_id[left_out_id] = [left_out_id]
                            else:
                                print("New ID added: ", left_out_id)
                                final_fuse_id[left_out_id] = [left_out_id]
            print("REID time: ", time() - t1)

            t1 = time()
            print('Final ids and their sub-ids:', final_fuse_id)
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for current_ids in ids_per_frame:
                        for f in current_ids:
                            #print('frame {} f0 {}'.format(frame,f[0]))
                            if str(i) == str(f) or str(idx) == str(f):
                                text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                                idx = int(idx[idx.find('_') + 1: :])
                                detection_track = track_cnt[f][0]
                                print("ID Matched: ", f)
                                cv2_addBox(idx, frame, detection_track[1], detection_track[2], detection_track[3], detection_track[4], line_thickness, text_thickness, text_scale)         
            print("Fusion time: ", time() - t1)

            del ids_per_frame[:]

            t1 = time()
            cv2.imshow("Device-{}".format(device), frame)
            print("Frame display time: ", time() - t1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        global stop_threads
        stop_threads = True
        cap.release()

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame, on each stream
        and displays the results
        :return: void
        """
        global feats
        self.threads = {}
        #Creating detection threads
        for d, u in self.devices.items():
            self.threads[d] = threading.Thread(target=self.detection_on_stream, args=(d, u))
        #Create one thread for extracting features
        self.threads["extract"] = threading.Thread(target=self._extract_features, args=(feats, self.feats_queue))
        #Starting detection threads
        for d in self.threads:
            self.threads[d].start()
        
        print("-------------Threads Running--------------------")

        for d in self.threads:
            self.threads[d].join()
        print("-------------Threads Stopped--------------------")

# deep_sort 
model_filename = 'model_data/models/mars-small128.pb'
from tools import generate_detections as gdet
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

#Definition of the parameters
max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 0.4
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=100)

reid = REID()
threshold = 320
exist_ids = set()
images_by_id = dict()
final_fuse_id = dict()
feats = dict()
stop_threads = False

# Create a new object and execute.
#, 2: "192.168.43.112:8080"
detector = ObjectDetection(capture_index=0, devices={1: "192.168.255.50:8080", 2: "192.168.255.112:8080"})
#detector = ObjectDetection(capture_index=0, devices={1: "192.168.255.50:8080"})
detector()