#pip install prefetch_generator

# 新建DataLoaderX类
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataPrefetcher():
    def __init__(self, loader, device='auto'):
        if device == 'auto':self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:raise NotImplementedError
        self.loader = iter(loader)

        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):

            for k in range(len(self.batch)):
                if isinstance(self.batch[k],list):
                    for i in range(len(self.batch[k])):
                        self.batch[k][i] = self.batch[k][i].float().to(device=self.device, non_blocking=True)
                else:
                    self.batch[k] = self.batch[k].float().to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
