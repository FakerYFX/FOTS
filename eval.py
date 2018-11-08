import argparse
import torch
import logging
import pathlib
import traceback
from models.FOTS import FOTS
from utils.bbox import Toolbox
import os

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path)
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    FOTS_model = FOTS()
    #FOTS_model = torch.nn.DataParallel(FOTS_model)
    FOTS_model.load_state_dict(checkpoints)
    if with_gpu:
        FOTS_model = FOTS_model.cuda()
    return FOTS_model


def main(args:argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False

    model = load_model(model_path, with_gpu)

    for image_fn in os.listdir(input_dir):
        try:
            with torch.no_grad():
                ploy, im = Toolbox.predict(image_fn, input_dir,model, with_image, output_dir, with_gpu)
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default="save_model/model_5.pth", type=str,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default="test_result/", type=str,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default="test_pic/", type=str, required=False,
                        help='dir for input images')
    args = parser.parse_args()
    main(args)










