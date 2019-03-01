from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import sys
import json
import yaml
import shutil
import logging
import threading
import numpy as np
import detectron.utils.c2 as c2_utils
import detectron.utils.train

from caffe2.python import workspace
from os import path as osp
from pprint import pprint
from pprint import pformat

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
from detectron.utils.convert_datasets_to_voc import convert
from detectron.utils.vocconverter import readCategoryFromJson
from detectron.utils.train import train_model

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)

class trainThread(threading.Thread):
    def __init__(self,jsondata, detectron_path):
        super(trainThread,self).__init__()
        self.jsondata=jsondata
        self.detectron=detectron_path

    def parseJsonToCFG(self, cfg):
        merge_cfg_from_file(self.jsondata["parameterPath"])
        para_dict = self.jsondata
        model_dict = para_dict.pop("modelParameter")
        solver_l = ["base_lr", "gamma", "max_iter", "steps"]
        train_l = ["scales", "max_size"]
        test_l = ["scale", "max_size", "nms"]
        parameter_l = ["TRAIN.DATASETS",("voc_2007_train",),  "TEST.DATASETS", ("voc_2007_val",)]
        for key in model_dict:
            if key in solver_l:
                parameter_l.extend(["SOLVER.%s"%key.upper(), model_dict[key]])
            if key in train_l:
                parameter_l.extend(["TRAIN.%s"%key.upper(), model_dict[key]])
            if key in test_l:
                parameter_l.extend(["TEST.%s"%key.upper(), model_dict[key]])
        parameter_l.extend(["OUTPUT_DIR", para_dict["model_Path"]])
        merge_cfg_from_list(parameter_l)
        

    def checkPath(self, sample):
        if not (osp.exists(osp.join(sample, "images")) 
        and osp.exists(osp.join(sample, "labels"))):
            print("Please check datasets at %s!"%sample)
            sys.exit()
        elif osp.exists(osp.join(sample, "VOC2007")):
            shutil.rmtree(osp.join(sample, "VOC2007"))
            print("Clean datasets dir.")

    def checkSymLink(self):
        link_path = osp.join(self.detectron, "detectron", "datasets", "data", "VOC2007")
        if osp.islink(link_path):
            os.unlink(link_path)
        os.symlink(osp.join(self.jsondata["sample"], "VOC2007"), link_path)

    def beginTrain(self):
        workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
        )
        # Set up logging and load config options
        logger = setup_logging(__name__)
        logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
        smi_output, cuda_ver, cudnn_ver = c2_utils.get_nvidia_info()
        logger.info("cuda version : {}".format(cuda_ver))
        logger.info("cudnn version: {}".format(cudnn_ver))
        logger.info("nvidia-smi output:\n{}".format(smi_output))
        logger.info('Training with config:')
        logger.info(pformat(cfg))
        # Note that while we set the numpy random seed network training will not be
        # deterministic in general. There are sources of non-determinism that cannot
        # be removed with a reasonble execution-speed tradeoff (such as certain
        # non-deterministic cudnn functions).
        np.random.seed(cfg.RNG_SEED)
        # Execute the training run
        checkpoints, losses = train_model()
        # Test the trained model
        self.test_model(checkpoints["final"])

    def test_model(self, model_file):
        """Test a model."""
        # Clear memory before inference
        workspace.ResetWorkspace()
        # Run inference
        run_inference(
            model_file,
            check_expected_results=True,
        )

    def run(self):
        sample = self.jsondata["sample"] 
        self.parseJsonToCFG(cfg)
        self.checkPath(sample)
        anno_path = convert(sample)
        categories = readCategoryFromJson(osp.join(anno_path, "voc_2007_train.json"))
        parameter_l = ["MODEL.NUM_CLASSES", len(categories)+1]
        merge_cfg_from_list(parameter_l)
        assert_and_infer_cfg()
        if not osp.exists(self.jsondata["model_Path"]):
            os.mkdir(self.jsondata["model_Path"])
        with open(osp.join(self.jsondata["model_Path"], "model.yaml"), "w") as src:
            src.write(yaml.dump(cfg))
        with open(osp.join(self.jsondata["model_Path"], "classes.txt"), "w") as src:
            for category in categories:
                src.write(category)
        self.checkSymLink()
        self.beginTrain()

if __name__ == "__main__":
    json_str = {"jobId":"", 
                "jobName":"", 
                "sample":"/home/szl/yanpan/Test/", 
                "parameterPath":"/home/szl/yanpan/detectron_ADC/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml", 
                "modelParameter":{
                    "steps":[0,],
                    "max_iter":60000, 
                    "base_lr":0.01, 
                    "gamma":0.1, 
                    "nms":0.5, 
                    "scales":(800,), 
                    "scale":800, 
                    "max_size":1333,  
                }, 
                "model_Path":"output_test"}
    detectron_path = "/home/szl/yanpan/detectron_ADC"
    thread = trainThread(json_str, detectron_path)
    thread.run()