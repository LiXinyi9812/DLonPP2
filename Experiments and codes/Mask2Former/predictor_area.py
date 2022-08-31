# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
from torchvision import transforms

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from PIL import Image
import PIL
import numpy as np
import os
import shutil
import zipfile


MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = [
    {'color': [165, 42, 42],
    'id': 1,
    'isthing': 1,
    'name': 'Bird',
    'supercategory': 'animal--bird'},
    {'color': [0, 192, 0],
    'id': 2,
    'isthing': 1,
    'name': 'Ground Animal',
    'supercategory': 'animal--ground-animal'},
    {'color': [196, 196, 196],
    'id': 3,
    'isthing': 0,
    'name': 'Curb',
    'supercategory': 'construction--barrier--curb'},
    {'color': [190, 153, 153],
    'id': 4,
    'isthing': 0,
    'name': 'Fence',
    'supercategory': 'construction--barrier--fence'},
    {'color': [180, 165, 180],
    'id': 5,
    'isthing': 0,
    'name': 'Guard Rail',
    'supercategory': 'construction--barrier--guard-rail'},
    {'color': [90, 120, 150],
    'id': 6,
    'isthing': 0,
    'name': 'Barrier',
    'supercategory': 'construction--barrier--other-barrier'},
    {'color': [102, 102, 156],
    'id': 7,
    'isthing': 0,
    'name': 'Wall',
    'supercategory': 'construction--barrier--wall'},
    {'color': [128, 64, 255],
    'id': 8,
    'isthing': 0,
    'name': 'Bike Lane',
    'supercategory': 'construction--flat--bike-lane'},
    {'color': [140, 140, 200],
    'id': 9,
    'isthing': 1,
    'name': 'Crosswalk - Plain',
    'supercategory': 'construction--flat--crosswalk-plain'},
    {'color': [170, 170, 170],
    'id': 10,
    'isthing': 0,
    'name': 'Curb Cut',
    'supercategory': 'construction--flat--curb-cut'},
    {'color': [250, 170, 160],
    'id': 11,
    'isthing': 0,
    'name': 'Parking',
    'supercategory': 'construction--flat--parking'},
    {'color': [96, 96, 96],
    'id': 12,
    'isthing': 0,
    'name': 'Pedestrian Area',
    'supercategory': 'construction--flat--pedestrian-area'},
    {'color': [230, 150, 140],
    'id': 13,
    'isthing': 0,
    'name': 'Rail Track',
    'supercategory': 'construction--flat--rail-track'},
    {'color': [128, 64, 128],
    'id': 14,
    'isthing': 0,
    'name': 'Road',
    'supercategory': 'construction--flat--road'},
    {'color': [110, 110, 110],
    'id': 15,
    'isthing': 0,
    'name': 'Service Lane',
    'supercategory': 'construction--flat--service-lane'},
    {'color': [244, 35, 232],
    'id': 16,
    'isthing': 0,
    'name': 'Sidewalk',
    'supercategory': 'construction--flat--sidewalk'},
    {'color': [150, 100, 100],
    'id': 17,
    'isthing': 0,
    'name': 'Bridge',
    'supercategory': 'construction--structure--bridge'},
    {'color': [70, 70, 70],
    'id': 18,
    'isthing': 0,
    'name': 'Building',
    'supercategory': 'construction--structure--building'},
    {'color': [150, 120, 90],
    'id': 19,
    'isthing': 0,
    'name': 'Tunnel',
    'supercategory': 'construction--structure--tunnel'},
    {'color': [220, 20, 60],
    'id': 20,
    'isthing': 1,
    'name': 'Person',
    'supercategory': 'human--person'},
    {'color': [255, 0, 0],
    'id': 21,
    'isthing': 1,
    'name': 'Bicyclist',
    'supercategory': 'human--rider--bicyclist'},
    {'color': [255, 0, 100],
    'id': 22,
    'isthing': 1,
    'name': 'Motorcyclist',
    'supercategory': 'human--rider--motorcyclist'},
    {'color': [255, 0, 200],
    'id': 23,
    'isthing': 1,
    'name': 'Other Rider',
    'supercategory': 'human--rider--other-rider'},
    {'color': [200, 128, 128],
    'id': 24,
    'isthing': 1,
    'name': 'Lane Marking - Crosswalk',
    'supercategory': 'marking--crosswalk-zebra'},
    {'color': [255, 255, 255],
    'id': 25,
    'isthing': 0,
    'name': 'Lane Marking - General',
    'supercategory': 'marking--general'},
    {'color': [64, 170, 64],
    'id': 26,
    'isthing': 0,
    'name': 'Mountain',
    'supercategory': 'nature--mountain'},
    {'color': [230, 160, 50],
    'id': 27,
    'isthing': 0,
    'name': 'Sand',
    'supercategory': 'nature--sand'},
    {'color': [70, 130, 180],
    'id': 28,
    'isthing': 0,
    'name': 'Sky',
    'supercategory': 'nature--sky'},
    {'color': [190, 255, 255],
    'id': 29,
    'isthing': 0,
    'name': 'Snow',
    'supercategory': 'nature--snow'},
    {'color': [152, 251, 152],
    'id': 30,
    'isthing': 0,
    'name': 'Terrain',
    'supercategory': 'nature--terrain'},
    {'color': [107, 142, 35],
    'id': 31,
    'isthing': 0,
    'name': 'Vegetation',
    'supercategory': 'nature--vegetation'},
    {'color': [0, 170, 30],
    'id': 32,
    'isthing': 0,
    'name': 'Water',
    'supercategory': 'nature--water'},
    {'color': [255, 255, 128],
    'id': 33,
    'isthing': 1,
    'name': 'Banner',
    'supercategory': 'object--banner'},
    {'color': [250, 0, 30],
    'id': 34,
    'isthing': 1,
    'name': 'Bench',
    'supercategory': 'object--bench'},
    {'color': [100, 140, 180],
    'id': 35,
    'isthing': 1,
    'name': 'Bike Rack',
    'supercategory': 'object--bike-rack'},
    {'color': [220, 220, 220],
    'id': 36,
    'isthing': 1,
    'name': 'Billboard',
    'supercategory': 'object--billboard'},
    {'color': [220, 128, 128],
    'id': 37,
    'isthing': 1,
    'name': 'Catch Basin',
    'supercategory': 'object--catch-basin'},
    {'color': [222, 40, 40],
    'id': 38,
    'isthing': 1,
    'name': 'CCTV Camera',
    'supercategory': 'object--cctv-camera'},
    {'color': [100, 170, 30],
    'id': 39,
    'isthing': 1,
    'name': 'Fire Hydrant',
    'supercategory': 'object--fire-hydrant'},
    {'color': [40, 40, 40],
    'id': 40,
    'isthing': 1,
    'name': 'Junction Box',
    'supercategory': 'object--junction-box'},
    {'color': [33, 33, 33],
    'id': 41,
    'isthing': 1,
    'name': 'Mailbox',
    'supercategory': 'object--mailbox'},
    {'color': [100, 128, 160],
    'id': 42,
    'isthing': 1,
    'name': 'Manhole',
    'supercategory': 'object--manhole'},
    {'color': [142, 0, 0],
    'id': 43,
    'isthing': 1,
    'name': 'Phone Booth',
    'supercategory': 'object--phone-booth'},
    {'color': [70, 100, 150],
    'id': 44,
    'isthing': 0,
    'name': 'Pothole',
    'supercategory': 'object--pothole'},
    {'color': [210, 170, 100],
    'id': 45,
    'isthing': 1,
    'name': 'Street Light',
    'supercategory': 'object--street-light'},
    {'color': [153, 153, 153],
    'id': 46,
    'isthing': 1,
    'name': 'Pole',
    'supercategory': 'object--support--pole'},
    {'color': [128, 128, 128],
    'id': 47,
    'isthing': 1,
    'name': 'Traffic Sign Frame',
    'supercategory': 'object--support--traffic-sign-frame'},
    {'color': [0, 0, 80],
    'id': 48,
    'isthing': 1,
    'name': 'Utility Pole',
    'supercategory': 'object--support--utility-pole'},
    {'color': [250, 170, 30],
    'id': 49,
    'isthing': 1,
    'name': 'Traffic Light',
    'supercategory': 'object--traffic-light'},
    {'color': [192, 192, 192],
    'id': 50,
    'isthing': 1,
    'name': 'Traffic Sign (Back)',
    'supercategory': 'object--traffic-sign--back'},
    {'color': [220, 220, 0],
    'id': 51,
    'isthing': 1,
    'name': 'Traffic Sign (Front)',
    'supercategory': 'object--traffic-sign--front'},
    {'color': [140, 140, 20],
    'id': 52,
    'isthing': 1,
    'name': 'Trash Can',
    'supercategory': 'object--trash-can'},
    {'color': [119, 11, 32],
    'id': 53,
    'isthing': 1,
    'name': 'Bicycle',
    'supercategory': 'object--vehicle--bicycle'},
    {'color': [150, 0, 255],
    'id': 54,
    'isthing': 1,
    'name': 'Boat',
    'supercategory': 'object--vehicle--boat'},
    {'color': [0, 60, 100],
    'id': 55,
    'isthing': 1,
    'name': 'Bus',
    'supercategory': 'object--vehicle--bus'},
    {'color': [0, 0, 142],
    'id': 56,
    'isthing': 1,
    'name': 'Car',
    'supercategory': 'object--vehicle--car'},
    {'color': [0, 0, 90],
    'id': 57,
    'isthing': 1,
    'name': 'Caravan',
    'supercategory': 'object--vehicle--caravan'},
    {'color': [0, 0, 230],
    'id': 58,
    'isthing': 1,
    'name': 'Motorcycle',
    'supercategory': 'object--vehicle--motorcycle'},
    {'color': [0, 80, 100],
    'id': 59,
    'isthing': 0,
    'name': 'On Rails',
    'supercategory': 'object--vehicle--on-rails'},
    {'color': [128, 64, 64],
    'id': 60,
    'isthing': 1,
    'name': 'Other Vehicle',
    'supercategory': 'object--vehicle--other-vehicle'},
    {'color': [0, 0, 110],
    'id': 61,
    'isthing': 1,
    'name': 'Trailer',
    'supercategory': 'object--vehicle--trailer'},
    {'color': [0, 0, 70],
    'id': 62,
    'isthing': 1,
    'name': 'Truck',
    'supercategory': 'object--vehicle--truck'},
    {'color': [0, 0, 192],
    'id': 63,
    'isthing': 1,
    'name': 'Wheeled Slow',
    'supercategory': 'object--vehicle--wheeled-slow'},
    {'color': [32, 32, 32],
    'id': 64,
    'isthing': 0,
    'name': 'Car Mount',
    'supercategory': 'void--car-mount'},
    {'color': [120, 10, 10],
    'id': 65,
    'isthing': 0,
    'name': 'Ego Vehicle',
    'supercategory': 'void--ego-vehicle'}
]
class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)



    def run_on_image(self, image, filename,output_dir_pt):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """

        def get_area(panoptic_seg, segments_info):
            area = torch.zeros(66)
            print(len(panoptic_seg.reshape(-1)))
            for id in panoptic_seg.reshape(-1):
                if id != 0:
                    area[segments_info[id - 1]['category_id']] += 1
                else:
                    area[0] += 1
            print('no divided area',area)
            area = torch.div(area,50176)
            return area

        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            info = torch.zeros(66)
            area = get_area(panoptic_seg,segments_info)
            for item in segments_info:
                info[item['category_id']] += 1
            cp = {
                  'area': area,
                  'info':info
            }
            print('panoptic_seg',panoptic_seg)
            print('segments_info', segments_info)
            print('info', info)
            print('area', area)
            torch.save(cp, os.path.join(output_dir_pt, '{}.pt'.format(filename)))


            vis_output = visualizer.draw_panoptic_seg_predictions(
               panoptic_seg.to(self.cpu_device),segments_info
            )

        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                   predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
