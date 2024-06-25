import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
import torch
import numpy as np
import argparse
from PIL import Image
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, topk_accuracy, verify_checkpoint_dir, task_confusion
from model import CNN_TRX, tsa

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import video_reader
import random 
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


def main():
    accuracy_list = []
    for i in range(10):
        learner = Learner()
        start_time = time.time()
        accuracy = learner.test()
        end_time = time.time()
        print('Time Consumption: {}s'.format(end_time-start_time))
        accuracy_list.append(accuracy)
        torch.cuda.empty_cache()
    accuracy_list = np.array(accuracy_list)
    mean_values, std_values = np.mean(accuracy_list, axis=0), np.std(accuracy_list, axis=0, ddof=1)
    print('Top-1: {}+/-{}, Top-2: {}+/-{}, Top-3: {}+/-{}, Top-5: {}+/-{}'.format(mean_values[0], std_values[0], 
                                                                                  mean_values[1], std_values[1], 
                                                                                  mean_values[2], std_values[2],
                                                                                  mean_values[3], std_values[3]))


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, True)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.writer = SummaryWriter()
        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        
        self.loss = loss
        self.accuracy_fn = topk_accuracy # aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_final_model()
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN_TRX(self.args)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf", "hri", "coax", "ourtoy"], default="ourtoy", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2,3])
        parser.add_argument("--scratch", choices=["bc", "bp"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument('--modality', nargs='+', type=str, help='modalities to use', default=['rgb'])


        args = parser.parse_args()
        
        if args.scratch == "bc":
            args.scratch = "./"
        elif args.scratch == "bp":
            args.num_gpus = 4
            args.num_workers = 3
            args.scratch = "./"
        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if (args.method == "resnet50") or (args.method == "resnet34"):
            args.img_size = 224
        if args.method == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512
        
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/somethingsomethingv2TrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/somethingsomethingv2_256x256q5_7l8.zip")
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/kineticsTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/kinetics_256q5_1.zip")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "trx/splits/ucf_ARN")
            args.path = os.path.join(args.scratch, "video_datasets/data/ucf101")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/hmdb51TrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_256q5.zip")
        elif args.dataset == "hri":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/hriTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/hri30")
        elif args.dataset == "coax":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/coaxTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/coax")
        elif args.dataset == "ourtoy":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/ourtoyTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/ourtoy")

        return args

    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i

    def get_seq(self, paths):
        n_frames = len(paths)
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            start = 1
            end = n_frames - 2
    
            if end - start < self.args.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.args.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.args.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]

        '''
        set up transforms
        '''
        video_test_list = []
        if self.args.img_size == 84:
            video_test_list.append(Resize(96))
        elif self.args.img_size == 224:
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_test_list.append(CenterCrop(self.args.img_size))
        transform = Compose(video_test_list)
        imgs = [transforms.ToTensor()(v) for v in transform(imgs)]
        imgs = torch.stack(imgs)
        return imgs

    def test(self):
        test_dir = self.args.path
        self.model.eval()
        self.model.add_adaptor()
        self.model.reset_adaptor()
        self.model = self.model.to(self.device) 
        
        num_way, num_shot = self.args.way, self.args.shot
        accuracy_dict = {}
        accuracies = []
        iteration = 0
        item = self.args.dataset
        support_set, support_labels = [], []
        target_set, target_labels, real_target_labels = [], [], []

        for class_ind, bc in enumerate(sorted(random.sample(os.listdir(test_dir), num_way))):
            for _, sample in enumerate(random.sample(os.listdir(os.path.join(test_dir, bc)), len(os.listdir(os.path.join(test_dir, bc))))):
                sample_dir = os.path.join(test_dir, bc, sample, 'rgb')  # 'rgb' for coax
                sample_dir = sorted(os.listdir(sample_dir))
                image_paths = [os.path.join(test_dir, bc, sample, 'rgb', img) for img in sample_dir]   # 'rgb' for coax
                vid  = self.get_seq(image_paths)
                if _ < num_shot:
                    support_set.append(vid)
                    support_labels.append(class_ind)
                elif _ < 25:
                    target_set.append(vid)
                    target_labels.append(class_ind)
                    real_target_labels.append(bc)

        support_set = torch.cat(support_set).to(self.device)
        target_set = torch.cat(target_set).to(self.device)
        support_labels = torch.tensor(support_labels).to(self.device)
        target_labels = torch.tensor(target_labels).type(torch.LongTensor).to(self.device)
        
        ft_start_time = time.time()
        tsa(support_set, support_labels, target_set, self.model, max_iter=100, lr_alpha=1e-4, lr_beta=1e-4, lr=1e-3) # 1e-4, 1e-4, 1e-3
        ft_end_time = time.time()
        print(ft_end_time-ft_start_time)

        # print(self.model.transformers[0].ia3.weight_k)
        with torch.no_grad():
            model_dict = self.model(support_set, support_labels, target_set)
            target_logits = model_dict['logits']
            accuracy = self.accuracy_fn(target_logits.squeeze(0), target_labels, num_classes=num_way, device=self.device)
            # print(target_logits, target_labels)
            accuracies.append(accuracy[0].item())
            accuracies.append(accuracy[1].item())
            accuracies.append(accuracy[2].item())
            accuracies.append(accuracy[3].item())
            print(accuracies)
        
        return accuracies


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        print('loading intermediate checkpoint...')
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def load_final_model(self):
        print('loading fully trained model...')
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'fully_trained.pt'))
        self.model.load_state_dict(checkpoint)


if __name__ == "__main__":
    main()
