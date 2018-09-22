from __future__ import division
import time


from yolo.darknet import Darknet
from yolo.preprocess import prep_image, inp_to_image


from yolo.util import *
import argparse


CUDA = torch.cuda.is_available()


def init_model(args):
    scales = args.scales

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    num_classes = 80
    classes = load_classes('yolo/data/coco.names')
    print("classes")
    print(classes)

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()
    return model

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help =
    "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =
    "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 32)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
    "Config file",
                        default = "yolo/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
    "weightsfile",
                        default = "yolo/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)

    return parser.parse_args()


class YoloEstimator:
    def __init__(self, args=arg_parse()):
        model = init_model(args)
        self.args = args
        self.model = model
        self.inp_dim = int(self.model.net_info["height"])
        self.batch_size = int(self.args.bs)
        self.confidence = float(args.confidence)
        self.nms_thesh = float(args.nms_thresh)
        self.num_classes = 80
        pass

    def predict(self, imlist):
        batches = list(
            map(prep_image, imlist, [self.inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if CUDA:
            im_dim_list = im_dim_list.cuda()

        leftover = 0

        if (len(im_dim_list) % self.batch_size):
            leftover = 1

        if self.batch_size != 1:
            num_batches = len(imlist) // self.batch_size + leftover
            im_batches = [
                torch.cat((im_batches[i * self.batch_size: min((i + 1) * self.batch_size,
                                                          len(im_batches))]))
                for i in range(num_batches)]

        i = 0
        write = False
        objs = {}
        for batch in im_batches:
            # load the image
            start = time.time()
            if CUDA:
                batch = batch.cuda()

            with torch.no_grad():
                prediction = self.model(Variable(batch), CUDA)

            prediction = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
            if type(prediction) == int:
                i += 1
                continue
            end = time.time()
            prediction[:, 0] += i * self.batch_size

            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output, prediction))

            i += 1

            if CUDA:
                torch.cuda.synchronize()

        try:
            output
        except NameError:
            print("No detections were made")
            exit()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(self.inp_dim / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(
            -1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(
            -1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                            im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                            im_dim_list[i, 1])

        output = output.detach().cpu().numpy()

        # collect result
        bbox_res = []
        for i in range(len(imlist)):
            cur_idx = np.where(output[:, 0].astype(np.int) == i)
            cur_output = output[cur_idx]
            human_idx = np.where(cur_output[:, -1].astype(np.int) == 0)
            if len(human_idx) == 1 and len(human_idx[0]) == 0:
                bbox_res.append([])
                print("error not found bbox")
                continue
            cur_output = cur_output[human_idx]
            max_idx = np.argmax(cur_output[:, 5] + cur_output[:, 6])
            # print(max_idx)
            # fout.write(basename(imlist[i]) + "," + ",".join(
            #     str(v) for v in list(cur_output[max_idx, 1:5])))
            # fout.write("\n")
            bbox_res.append(cur_output[max_idx, 1:5])
        return bbox_res


