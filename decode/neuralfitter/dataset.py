import time

import torch
from torch.utils.data import Dataset

from decode.generic import emitter
import decode.simulation 
import numpy as np 


class SMLMDataset(Dataset):
    """
    SMLM base dataset.


    """
    _pad_modes = (None, 'same')

    def __init__(self, *, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen,
                 frame_window: int, pad: str = None, return_em: bool):
        """
        Init new dataset.

        Args:
            em_proc: Emitter processing
            frame_proc: Frame processing
            bg_frame_proc: Background frame processing
            tar_gen: Target generator
            weight_gen: Weight generator
            frame_window: number of frames per sample / size of frame window
            pad: pad mode, applicable for first few, last few frames (relevant when frame window is used)
            return_em: return target emitter

        """
        super().__init__()

        self._frames = None
        self._emitter = None

        self.em_proc = em_proc
        self.frame_proc = frame_proc
        self.bg_frame_proc = bg_frame_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen

        self.frame_window = frame_window
        self.pad = pad
        self.return_em = return_em

        """Sanity"""
        self.sanity_check()

    def __len__(self):
        if self.pad is None:  # loosing samples at the border
            return self._frames.size(0) - self.frame_window + 1

        elif self.pad == 'same':
            return self._frames.size(0)

    def sanity_check(self):
        """
        Checks the sanity of the dataset, if fails, errors are raised.

        """

        if self.pad not in self._pad_modes:
            raise ValueError(f"Pad mode {self.pad} not available. Available pad modes are {self._pad_modes}.")

        if self.frame_window is not None and self.frame_window % 2 != 1:
            raise ValueError(f"Unsupported frame window. Frame window must be odd integered, not {self.frame_window}.")

    def _get_frames(self, frames, index):
        hw = (self.frame_window - 1) // 2  # half window without centre

        frame_ix = torch.arange(index - hw, index + hw + 1).clamp(0, len(frames) - 1)
        return frames[frame_ix]

    def _pad_index(self, index):

        if self.pad is None:
            assert index >= 0, "Negative indexing not supported."
            return index + (self.frame_window - 1) // 2

        elif self.pad == 'same':
            return index

    def _process_sample(self, frames, tar_emitter, bg_frame):

        """Process"""
        if self.frame_proc is not None:
            frames = self.frame_proc.forward(frames)

        if self.bg_frame_proc is not None:
            bg_frame = self.bg_frame_proc.forward(bg_frame)

        if self.em_proc is not None:
            tar_emitter = self.em_proc.forward(tar_emitter)

        if self.tar_gen is not None:
            target = self.tar_gen.forward(tar_emitter, bg_frame)
        else:
            target = None

        if self.weight_gen is not None:
            weight = self.weight_gen.forward(tar_emitter, target)
        else:
            weight = None

        return frames, target, weight, tar_emitter

    def _return_sample(self, frame, target, weight, emitter):

        if self.return_em:
            return frame, target, weight, emitter
        else:
            return frame, target, weight


class SMLMStaticDataset(SMLMDataset):
    """
    A simple and static SMLMDataset.

    Attributes:
        frame_window (int): width of frame window

        tar_gen: target generator function
        frame_proc: frame processing function
        em_proc: emitter processing / filter function
        weight_gen: weight generator function

        return_em (bool): return EmitterSet in getitem method.
    """

    def __init__(self, *, frames, emitter: (None, list, tuple),
                 frame_proc=None, bg_frame_proc=None, em_proc=None, tar_gen=None,
                 bg_frames=None, weight_gen=None, frame_window=3, pad: (str, None) = None, return_em=True):
        """

        Args:
            frames (torch.Tensor): frames. N x H x W
            em (list of EmitterSets): ground-truth emitter-sets
            frame_proc: frame processing function
            em_proc: emitter processing / filter function
            tar_gen: target generator function
            weight_gen: weight generator function
            frame_window (int): width of frame window
            return_em (bool): return EmitterSet in getitem method.
        """

        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)

        self._frames = frames
        self._emitter = emitter
        self._bg_frames = bg_frames

        if self._frames is not None and self._frames.dim() != 3:
            raise ValueError("Frames must be 3 dimensional, i.e. N x H x W.")

        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("Please split emitters in list of emitters by their frame index first.")

    def __getitem__(self, ix):
        """
        Get a training sample.

        Args:
            ix (int): index

        Returns:
            frames (torch.Tensor): processed frames. C x H x W
            tar (torch.Tensor): target
            em_tar (optional): Ground truth emitters

        """

        """Pad index, get frames and emitters."""
        ix = self._pad_index(ix)

        tar_emitter = self._emitter[ix] if self._emitter is not None else None
        frames = self._get_frames(self._frames, ix)
        bg_frame = self._bg_frames[ix] if self._bg_frames is not None else None

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)

        return self._return_sample(frames, target, weight, tar_emitter)


class SMLMStaticDataset_simulations(SMLMStaticDataset): # changed type from SMLMDataset to SMLMStaticDataset so that we still have the original one
    """
    A simple and static SMLMDataset.

    Attributes:
        frame_window (int): width of frame window

        tar_gen: target generator function
        frame_proc: frame processing function
        em_proc: emitter processing / filter function
        weight_gen: weight generator function

        return_em (bool): return EmitterSet in getitem method.
    """

    def __init__(self, *, frames, emitter: (None, list, tuple),
                 frame_proc=None, bg_frame_proc=None, em_proc=None, tar_gen=None,
                 bg_frames=None, weight_gen=None, frame_window=3, pad: (str, None) = None, return_em=True):
        """

        Args:
            frames (torch.Tensor): frames. N x H x W
            em (list of EmitterSets): ground-truth emitter-sets # MB : emitter
            frame_proc: frame processing function
            em_proc: emitter processing / filter function
            tar_gen: target generator function
            weight_gen: weight generator function
            frame_window (int): width of frame window
            return_em (bool): return EmitterSet in getitem method.
        """

        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em,
                         frames = frames, emitter = emitter) # MB for static work
        # print("StaticDataset simulations!")
        self._frames = frames
        self._emitter = emitter
        self._bg_frames = bg_frames

        if self._frames is not None and self._frames.dim() != 3:
            raise ValueError("Frames must be 3 dimensional, i.e. N x H x W.")

        # if self._emitter is not None and not isinstance(self._emitter, (list, tuple)): # MB 
        #     print(self._emitter)
        #     raise TypeError("Please split emitters in list of emitters by their frame index first.")

    def __getitem__(self, ix):
        """
        Get a training sample.

        Args:
            ix (int): index

        Returns:
            frames (torch.Tensor): processed frames. C x H x W
            tar (torch.Tensor): target
            em_tar (optional): Ground truth emitters

        """

        """Pad index, get frames and emitters."""
        ix = self._pad_index(ix)

        tar_emitter = self._emitter[ix] if self._emitter is not None else None
        frames = self._get_frames(self._frames, ix)
        bg_frame = self._bg_frames[ix] if self._bg_frames is not None else None

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)
        # print("frames", frames.shape) # MB
        # print("frames", frames)
        # print("target0", target[0].shape)
        # print("target1", target[1].shape)
        # print("tar_emitter", tar_emitter)
        return self._return_sample(frames, target, weight, tar_emitter)

    def sample(self, verbose: bool = False, split: bool = True):
        """
        Sample new acquisition, i.e. a whole dataset.

        Args:
            verbose: print performance / verification information

        """ 

        def set_frame_ix(em):  # helper function
            em.frame_ix = torch.zeros_like(em.frame_ix)
            return em

        # """Sample new dataset."""
        # t0 = time.time()
        # emitter, frames, bg_frames = self.simulator.sample()
        # if verbose:
        #     print(f"Sampled dataset in {time.time() - t0:.2f}s. {len(emitter)} emitters on {frames.size(0)} frames.")

        # """Split Emitters into list of emitters (per frame) and set frame_ix to 0."""
        # emitter = emitter.split_in_frames(0, frames.size(0) - 1)
        # emitter = [set_frame_ix(em) for em in emitter]

        # self._emitter = emitter
        # self._frames = frames.cpu()
        # self._bg_frames = bg_frames.cpu()

        # """Sample new dataset -- MB."
        min_trajs, max_trajs = 10, 20
        traj_number = np.random.randint(min_trajs,max_trajs+1)  
        traj_len = 100
        frame_duration = 30
        dt_par = 5
        
        x, y = decode.simulation.movie_generator.simulate_BM(traj_len, 
            traj_number, frame_duration = frame_duration, dt = dt_par) 
            # simulate_BM(N, M, sigma=1.0, frame_duration = 30, dt = 5, corr = 0)
        pixel_number = 40 # param.Simulation.img_size[0] #param.HyperParameter.img_size[0]
        boundary_type = "square"

        X, Y = decode.simulation.movie_generator.simulate_trajectories(x, 
                                                                    y, 
                                                                    pixel_number, 
                                                                    boundary_type)
        
        # Yann's movie parameters
        background = 550 # level of background
        sigma_dye = 1.5 # sigma of PSF in pixel
        sigma_noise = 500 # level of gaussian noise on each image
        PSF_intensity = 3000 # maximum of the PSF
        dx = 1.0

        movie_param = decode.simulation.movie_generator.Movie_Parameters(background, sigma_dye, sigma_noise, PSF_intensity, dx)

        # parameters for saving        
        saved = 0
        direct_show = 0
        file_name = ""
        save_path = ""
        
        downscale_factor = 1


        stack, X_f, Y_f, _, _ = decode.simulation.movie_generator.simulate_movie(X, Y, pixel_number, boundary_type, saved, direct_show, 
            file_name, movie_param, save_path, downscale_factor = downscale_factor, frame_duration = frame_duration, dt = dt_par)
        
        em = decode.simulation.movie_generator.movie_to_emitter(X_f,Y_f,split)
        
        self._emitter = em
        self._frames = stack
        self._bg_frames = stack * 0 + background

        self.emitter = self._emitter

        # print("dataset.py bg", self._bg_frames) # MB
        # print("dataset bg size", self._bg_frames.size(0)) # MB
        # print("dataset.py frames mean ", torch.mean(self._bg_frames))

        # print("dataset.py frames", stack) # MB
        # print("dataset.py frames size", stack.size()) # MB
        # print("dataset.py frames min mean max ", torch.min(stack), torch.mean(stack), torch.max(stack))



class InferenceDataset(SMLMStaticDataset):
    """
    A SMLM dataset without ground truth data.
    This is dummy wrapper to keep the visual appearance of a separate dataset.
    """

    def __init__(self, *, frames, frame_proc, frame_window):
        """

        Args:
            frames (torch.Tensor): frames
            frame_proc: frame processing function
            frame_window (int): frame window
        """
        super().__init__(frames=frames, emitter=None, frame_proc=frame_proc, bg_frame_proc=None, em_proc=None,
                         tar_gen=None, pad='same', frame_window=frame_window, return_em=False)

    def _return_sample(self, frame, target, weight, emitter):
        return frame


class SMLMLiveDataset(SMLMStaticDataset):
    """
    A SMLM dataset where new datasets is sampleable via the sample() method of the simulation instance.
    The final processing on frame, emitters and target is done online.

    """

    def __init__(self, *, simulator, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, pad,
                 return_em=False):

        super().__init__(emitter=None, frames=None,
                         em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)

        self.simulator = simulator
        self._bg_frames = None

    def sanity_check(self):

        super().sanity_check()
        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("EmitterSet shall be stored in list format, where each list item is one target emitter.")

    def sample(self, verbose: bool = False):
        """
        Sample new acquisition, i.e. a whole dataset.

        Args:
            verbose: print performance / verification information

        """

        def set_frame_ix(em):  # helper function
            em.frame_ix = torch.zeros_like(em.frame_ix)
            return em

        """Sample new dataset."""
        t0 = time.time()
        emitter, frames, bg_frames = self.simulator.sample()
        if verbose:
            print(f"Sampled dataset in {time.time() - t0:.2f}s. {len(emitter)} emitters on {frames.size(0)} frames.")

        """Split Emitters into list of emitters (per frame) and set frame_ix to 0."""
        emitter = emitter.split_in_frames(0, frames.size(0) - 1)
        emitter = [set_frame_ix(em) for em in emitter]

        self._emitter = emitter
        self._frames = frames.cpu()
        self._bg_frames = bg_frames.cpu()


class SMLMAPrioriDataset(SMLMLiveDataset):
    """
    A SMLM Dataset where new data is sampled and processed in an 'a priori' manner, i.e. once per epoche. This is useful
    when processing is fast. Since everything is ready a few number of workers for the dataloader will suffice.

    """

    def __init__(self, *, simulator, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, pad,
                 return_em=False):
        super().__init__(simulator=simulator, em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen, frame_window=frame_window, pad=pad,
                         return_em=return_em)

        self._em_split = None  # emitter splitted in frames
        self._target = None
        self._weight = None

    @property
    def emitter(self) -> emitter.EmitterSet:
        """
        Return emitter with same indexing frames are returned; i.e. when pad same is used, the emitters frame index
        is not changed. When pad is None, the respective frame index is corrected for the frame window.

        """
        if self.pad == 'same':
            return self._emitter

        elif self.pad is None:
            hw = (self.frame_window - 1) // 2  # half window without centre

            # ToDo: Change here when pythonize emitter / frame indexing
            em = self._emitter.get_subset_frame(hw, len(self))
            em.frame_ix -= hw

            return em
        else:
            raise ValueError

    def sample(self, verbose: bool = False):
        """
        Sample new dataset and process them instantaneously.

        Args:
            verbose:

        """
        t0 = time.time()
        emitter, frames, bg_frames = self.simulator.sample()

        if verbose:
            print(f"Sampled dataset in {time.time() - t0:.2f}s. {len(emitter)} emitters on {frames.size(0)} frames.")

        frames, target, weight, tar_emitter = self._process_sample(frames, emitter, bg_frames)
        self._frames = frames.cpu()
        self._emitter = tar_emitter
        self._em_split = tar_emitter.split_in_frames(0, frames.size(0) - 1)
        self._target, self._weight = target, weight

    def __getitem__(self, ix):
        """

        Args:
            ix:

        Returns:

        """
        """Pad index, get frames and emitters."""
        ix = self._pad_index(ix)

        return self._return_sample(self._get_frames(self._frames, ix), [tar[ix] for tar in self._target],  # target is tuple
                                   self._weight[ix] if self._weight is not None else None,
                                   self._em_split[ix])


class SMLMLiveSampleDataset(SMLMDataset):
    """
    A SMLM dataset where a new sample is drawn per (training) sample.

    """

    def __init__(self, *, simulator, ds_len, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window,
                 return_em=False):
        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=None, return_em=return_em)

        self.simulator = simulator
        self.ds_len = ds_len

    def __len__(self):
        return self.ds_len

    def __getitem__(self, ix):
        """Sample"""
        emitter, frames, bg_frames = self.simulator.sample()

        assert frames.size(0) % 2 == 1
        frames = self._get_frames(frames, (frames.size(0) - 1) // 2)
        tar_emitter = emitter.get_subset_frame(0, 0)  # target emitters are the zero ones
        bg_frames = bg_frames[(self.frame_window - 1) // 2]  # ToDo: Beautify this

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frames)

        return self._return_sample(frames, target, weight, tar_emitter)
