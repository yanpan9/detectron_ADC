import sys
import yaml
import json
import pprint

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list

if __name__ == "__main__":
    para = sys.stdin.read()
    para_dict = json.loads(para)
    model_dict = para_dict.pop("modelParameter")
    solver_l = ["base_lr", "gamma", "max_iter", "steps"]
    train_l = ["scales", "max_size"]
    test_l = ["scale", "max_size", "nms"]
    parameter_l = list()
    for key in model_dict:
        if key in solver_l:
            parameter_l.extend(["SOLVER.%s"%key.upper(), model_dict[key]])
        if key in train_l:
            parameter_l.extend(["TRAIN.%s"%key.upper(), model_dict[key]])
        if key in test_l:
            parameter_l.extend(["TEST.%s"%key.upper(), model_dict[key]])
    parameter_l.extend(["OUTPUT_DIR", para_dict["model_Path"]])
    merge_cfg_from_list(parameter_l)
    assert_and_infer_cfg()
    with open("%s/model.yaml"%para_dict["model_Path"], "w") as src:
        src.write(yaml.dump(cfg))

    

        