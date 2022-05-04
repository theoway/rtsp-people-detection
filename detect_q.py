import multiprocessing as mp
from time import sleep, time
from tracemalloc import start
from turtle import left
from typing import Dict

import torch
import cv2
import numpy as np
import operator
import threading


model_filename = 'model_data/models/mars-small128.pb'
from tools import generate_detections as gdet
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

#Definition of the parameters
max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 0.4


from reid import REID

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

    def __init__(self, capture_index, feats_dict_shared, images_queue_shared):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.feats_dict_shared = feats_dict_shared
        self.images_queue_shared = images_queue_shared
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def box_transform(self, box: list, shape: tuple):
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


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        t1 = threading.Thread(target=self.proxy, args=(1, "192.168.231.96:8080", ))
        t2 = threading.Thread(target=self.proxy, args=(2, "192.168.231.96:8080", ))
        
        t1.start()
        t2.start()
    
        t1.join()
        t2.join()

    def proxy(self, device, url):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        from deep_sort.detection import Detection
        from deep_sort.tracker import Tracker
        from deep_sort import nn_matching
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_age=100)


        global images_by_id
        global threshold
        global exist_ids
        global final_fuse_id
        global reid
        global FeatsLock
        global QLock

        cap = cv2.VideoCapture()
        cap.open("http://{}/video".format(url))
        assert cap.isOpened()

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        id_prefix = str(device) + "_"
        
        images_by_id[device] = {}
        frame_cnt = 0
        while True:
            ids_per_frame = []
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.score_frame(frame)
            #frame = self.plot_boxes(results, frame)
            
            boxs = [self.box_transform(cords, (frame.shape[1], frame.shape[0])) for cords in results[1]] #[minx, miny, w, h]

            features = encoder(frame, boxs) # n * 128
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
            text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

            tracker.predict()
            tracker.update(detections)
            tmp_ids = []

            track_cnt = dict()
            frame_cnt += 1
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
                    idx = int(ids[ids.find('_') + 1: :])
                    #cv2_addBox(idx, frame, track_cnt[ids][0][1], track_cnt[ids][0][2], track_cnt[ids][0][3], track_cnt[ids][0][4], line_thickness, text_thickness, text_scale)
            if len(tmp_ids) > 0:
                ids_per_frame.append(set(tmp_ids))
            #print("IDs per frame: ", ids_per_frame)
            end_time = time()
        
            for i in images_by_id[device]:
                self.images_queue_shared.put([i, frame_cnt, images_by_id[device][i]])

            for f in ids_per_frame:
                if f:
                    if len(exist_ids) == 0:
                        for i in f:
                            final_fuse_id[i] = [i]
                        exist_ids = exist_ids or f
                    else:
                        #print("Exist IDs: ", exist_ids, "--------!!!!!!!!!!!!!!!!!!")
                        new_ids = f - exist_ids
                        for nid in new_ids:
                            dis = []
                            FeatsLock.acquire()
                            #print("Started collecting with NEW ids")
                            t = time()
                            if not nid in self.feats_dict_shared.keys() or self.feats_dict_shared[nid].shape[0] < 10:
                                exist_ids.add(nid)
                                if nid in self.feats_dict_shared.keys():
                                    print("Not enough feats: {}, ID: {}".format(self.feats_dict_shared[nid].shape[0], nid))
                                else:
                                    print("New ID to be extracted: {}".format(nid))
                                FeatsLock.release()
                                continue
                            else:
                                FeatsLock.release()
                            print("finished collecting with NEW ids: ", time() - t)
                        '''unpickable = []
                            for i in f:
                                for key,item in final_fuse_id.items():
                                    if i in item:
                                        unpickable += final_fuse_id[key]
                            print('----------------- exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                            for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                                tmp = np.mean(reid.compute_distance(self.feats_dict_shared[nid], self.feats_dict_shared[oid]))
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
                                final_fuse_id[nid] = [nid]'''

                        unpickable = []
                        for i in f:
                            for key,item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        for left_out_id in f & (exist_ids - set(unpickable)):
                            dis = []
                            FeatsLock.acquire()
                            #print("Started reiding with OLD ids")
                            t = time()
                            if not left_out_id in self.feats_dict_shared.keys() or self.feats_dict_shared[left_out_id].shape[0] < 10:
                                '''if left_out_id in self.feats_dict_shared.keys():
                                    print("Not enough feats: {}, ID: {}".format(self.feats_dict_shared[left_out_id].shape[0], left_out_id))
                                else:
                                    print("Leftout ID to be extracted: {}".format(left_out_id))'''
                                FeatsLock.release()
                                continue
                            for main_id in final_fuse_id.keys():
                                tmp = np.mean(reid.compute_distance(self.feats_dict_shared[left_out_id],self.feats_dict_shared[main_id]))
                                print('------------------- Left out {}, Main ID {}, tmp {}'.format(left_out_id, main_id, tmp))
                                dis.append([main_id, tmp])
                            FeatsLock.release()
                            print("finished reiding with OLD ids: ", time() - t)
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

            #print('Final ids and their sub-ids:', final_fuse_id)
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for current_ids in ids_per_frame:
                        for f in current_ids:
                            if str(i) == str(f) or str(idx) == str(f):
                                text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                                _idx = int(idx[idx.find('_') + 1: :])
                                detection_track = track_cnt[f][0]
                                #print("IDs Matched: ", f)
                                cv2_addBox(_idx, frame, detection_track[1], detection_track[2], detection_track[3], detection_track[4], line_thickness, text_thickness, text_scale)         



            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
            
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            
            cv2.imshow('Device: {}'.format(device), frame)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()

def extract_features(feats, q, f_lock, q_lock) -> None:
        from reid import REID
        reid = REID()
        print("Extracting process has started!!!!!!!!")
        l_dict = dict()
        while True:
            if not q.empty():
                id, cnt, img = q.get()

                if id in l_dict.keys():
                    if l_dict[id][0] < cnt:
                        l_dict[id] = [cnt, img]
                else:
                    l_dict[id] = [cnt, img]

                f = reid._features(l_dict[id][1])
                f_lock.acquire()
                t = time()
                feats[id] = f
                f_lock.release()
                print("Succesfully extracted feat ID: ", id, "\tTime taken: ", time() - t)
            
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    images_by_id = dict()

    threshold = 380
    reid = REID()

    exist_ids = set()
    final_fuse_id = dict()

    QLock = mp.Lock()
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()
    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock, QLock, ))
    extract_p.start()
    
    try:
        detector = ObjectDetection(0, shared_feats_dict, shared_images_queue)
        detector()
    except Exception as e:
        print("Error: ", e)
    finally:
        extract_p.terminate()
        extract_p.join()
        print("Keys total------------: ", len(shared_feats_dict.keys()), shared_feats_dict)
        shared_images_queue.close()