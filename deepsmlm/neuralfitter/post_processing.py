import numpy as np
from scipy.ndimage.measurements import label
# from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

from deepsmlm.generic.coordinate_trafo import UpsamplingTransformation as ScaleTrafo
from deepsmlm.generic.coordinate_trafo import A2BTransform
from deepsmlm.generic.psf_kernel import DeltaPSF, OffsetPSF
from deepsmlm.generic.emitter import EmitterSet


class PeakFinder:
    """
    Class to find a local peak of the network output.
    This is similiar to non maximum suppresion.
    """

    def __init__(self, threshold, min_distance, extent, upsampling_factor):
        """
        Documentation from http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max if
        parameters applicable to peak_local_max

        :param threshold: Minimum intensity of peaks. By default, the absolute threshold is the minimum intensity of the image.
        :param min_distance: Minimum number of pixels separating peaks in a region of 2 * min_distance + 1 (i.e. peaks
         are separated by at least min_distance). To find the maximum number of peaks, use min_distance=1.
        :param extent: extent of the input image
        :param upsampling_factor: factor by which input image is upsampled
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.extent = extent
        self.upsampling_factor = upsampling_factor
        self.transformation = ScaleTrafo(self.extent, self.upsampling_factor)

    def forward(self, img):
        """
        Forward img to find the peaks (a way of declustering).
        :param img: batchised image --> N x C x H x W
        :return: emitterset
        """
        if img.dim() != 4:
            raise ValueError("Wrong dimension of input image. Must be N x C=1 x H x W.")

        n_batch = img.shape[0]
        coord_batch = []
        img_ = img.detach().numpy()
        for i in range(n_batch):
            cord = np.ascontiguousarray(peak_local_max(img_[i, 0, :, :],
                                                       min_distance=self.min_distance,
                                                       threshold_abs=self.threshold,
                                                       exclude_border=False))

            cord = torch.from_numpy(cord)

            # Transform cord based on image to cord based on extent
            cord = self.transformation.up2coord(cord)
            coord_batch.append(EmitterSet(cord,
                                        (torch.ones(cord.shape[0]) * (-1)),
                                        frame_ix=torch.zeros(cord.shape[0])))

        return coord_batch


class CoordScan:
    """Cluster to coordinate midpoint post processor"""

    def __init__(self, cluster_dims, eps=0.5, phot_threshold=0.8, clusterer=None):

        self.cluster_dims = cluster_dims
        self.eps = eps
        self.phot_tr = phot_threshold

        if clusterer is None:
            self.clusterer = DBSCAN(eps=eps, min_samples=phot_threshold)

    def forward(self, xyz, phot):
        """
        Forward a batch of list of coordinates through the clustering algorithm.

        :param xyz: batchised coordinates (Batch x N x D)
        :param phot: batchised photons (Batch X N)
        :return: list of tensors of clusters, and list of tensor of photons
        """
        assert xyz.dim() == 3
        batch_size = xyz.shape[0]

        xyz_out = [None] * batch_size
        phot_out = [None] * batch_size

        """Loop over the batch"""
        for i in range(batch_size):
            xyz_ = xyz[i, :, :].numpy()
            phot_ = phot[i, :].numpy()

            if self.cluster_dims == 2:
                db = self.clusterer.fit(xyz_[:, :2], phot_)
            else:
                core_samples, clus_ix = self.clusterer.fit(xyz_, phot_)

            core_samples = db.core_sample_indices_
            clus_ix = db.labels_

            core_samples = torch.from_numpy(core_samples)
            clus_ix = torch.from_numpy(clus_ix)
            num_cluster = clus_ix.max() + 1  # because -1 means not in cluster, and then from 0 - max_ix

            xyz_batch_cluster = torch.zeros((num_cluster, xyz_.shape[1]))
            phot_batch_cluster = torch.zeros(num_cluster)

            """Loop over the clusters"""
            for j in range(num_cluster):
                in_clus = clus_ix == j

                xyz_clus = xyz_[in_clus, :]
                phot_clus = phot_[in_clus]

                """Calculate weighted average. Maybe replace by (weighted) median?"""
                clus_mean = np.average(xyz_clus, axis=0, weights=phot_clus)
                xyz_batch_cluster[j, :] = torch.from_numpy(clus_mean)
                photons = phot_clus.sum()
                phot_batch_cluster[j] = photons

            xyz_out[i] = xyz_batch_cluster
            phot_out[i] = phot_batch_cluster

        return xyz_out, phot_out


class ConnectedComponents:
    def __init__(self, svalue_th=0, connectivity=2):
        self.svalue_th = svalue_th
        self.clusterer = label

        if connectivity == 2:
            self.kernel = np.ones((3, 3))
        elif connectivity == 1:
            self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def compute_cix(self, p_map):
        """Computes cluster ix, based on a prob-map.

        :param p_map: either N x H x W or H x W
        :return: cluster indices (NHW or HW)
        """
        if p_map.dim() == 2:
            out_hw = True  # output in hw format, i.e. without N
            p = p_map.clone().unsqueeze(0)
        else:
            out_hw = False  # output in NHW format
            p = p_map.clone()

        """Set all values under the single value threshold to 0."""
        p[p < self.svalue_th] = 0.

        cluster_ix = torch.zeros_like(p)

        for i in range(p.size(0)):
            c_ix, _ = label(p[i].numpy(), self.kernel)
            cluster_ix[i] = torch.from_numpy(c_ix)

        if out_hw:
            return cluster_ix.squeeze(0)
        else:
            return cluster_ix


class SpeiserPost:

    def __init__(self, svalue_th=0.3, sep_th=0.6, out_format='emitters', out_th=0.7):
        """

        :param svalue_th: single value threshold
        :param sep_th: threshold when to assume that we have 2 emitters
        :param out_format: either 'emitters' or 'image'. If 'emitter' we output instance of EmitterSet, if 'frames' we output post_processed frames.
        :param out_th: final threshold
        """
        self.svalue_th = svalue_th
        self.sep_th = sep_th
        self.out_format = out_format
        self.out_th = out_th

    def forward_(self, p, features):
        """
        :param p: N x H x W probability map
        :param features: N x C x H x W features
        :return: feature averages N x (1 + C) x H x W final probabilities plus features
        """
        with torch.no_grad():
            diag = 0
            p_ = p.clone()
            features = features.clone()

            # probability values > 0.3 are regarded as possible locations
            p_clip = torch.where(p > self.svalue_th, p, torch.zeros_like(p))[:, None]

            # localize maximum values within a 3x3 patch
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            # Add probability values from the 4 adjacent pixels
            filt = torch.tensor([[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]).view(1, 1, 3, 3).\
                type(features.dtype).\
                to(features.device)
            conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1)
            p_ps1 = max_mask1 * conv

            """In order do be able to identify two fluorophores in adjacent pixels we look for 
            probablity values > 0.6 that are not part of the first mask."""
            p_ *= (1 - max_mask1[:, 0])
            p_clip = torch.where(p_ > self.sep_th, p_, torch.zeros_like(p_))[:, None]
            max_mask2 = torch.where(p_ > self.sep_th, torch.ones_like(p_), torch.zeros_like(p_))[:, None]
            p_ps2 = max_mask2 * conv

            """This is our final clustered probablity which we then threshold (normally > 0.7) 
            to get our final discrete locations."""
            p_ps = p_ps1 + p_ps2

            max_mask = torch.clamp(max_mask1 + max_mask2, 0, 1)

            mult_1 = max_mask1 / p_ps1
            mult_1[torch.isnan(mult_1)] = 0
            mult_2 = max_mask2 / p_ps2
            mult_2[torch.isnan(mult_2)] = 0

            feat_out = torch.zeros_like(features)
            for i in range(features.size(1)):
                feature_mid = features[:, i] * p
                feat_conv1 = torch.nn.functional.conv2d((feature_mid * (1 - max_mask2[:, 0]))[:, None], filt, padding=1)
                feat_conv2 = torch.nn.functional.conv2d((feature_mid * (1 - max_mask1[:, 0]))[:, None], filt, padding=1)

                feat_out[:, [i]] = feat_conv1 * mult_1 + feat_conv2 * mult_2

            feat_out[torch.isnan(feat_out)] = 0

        """Output """
        combined_output = torch.cat((p_ps, feat_out), dim=1)

        return combined_output

    def forward(self, features):
        """
        Wrapper method calling forward_masked which is the actual implementation.

        :param features: NCHW
        :return: feature averages N x C x H x W if self.out_format == frames,
            list of EmitterSets if self.out_format == 'emitters'
        """
        post_frames = self.forward_(features[:, 0], features[:, 1:])
        is_above_out_th = (post_frames[:, [0], :, :] > self.out_th)

        post_frames = post_frames * is_above_out_th.type(post_frames.dtype)
        batch_size = post_frames.shape[0]

        """Output according to format as specified."""
        if self.out_format == 'frames':
            return post_frames

        elif self.out_format == 'emitters':
            is_above_out_th.squeeze_(1)
            frame_ix = torch.ones_like(post_frames[:, 0, :, :]) * \
                       torch.arange(batch_size, dtype=post_frames.dtype).view(-1, 1, 1, 1)
            frame_ix = frame_ix[:, 0, :, :][is_above_out_th]
            p_map = post_frames[:, 0, :, :][is_above_out_th]
            phot_map = post_frames[:, 1, :, :][is_above_out_th]
            x_map = post_frames[:, 2, :, :][is_above_out_th]
            y_map = post_frames[:, 3, :, :][is_above_out_th]
            z_map = post_frames[:, 4, :, :][is_above_out_th]
            xyz = torch.cat((
                x_map.unsqueeze(1),
                y_map.unsqueeze(1),
                z_map.unsqueeze(1)
            ), 1)
            em = EmitterSet(xyz, phot_map, frame_ix)
            em.p = p_map
            return em.split_in_frames(0, batch_size - 1)


def speis_post_functional(x):
    """
    A dummy wrapper because I don't get tracing to work otherwise.

    :param features: N x C x H x W
    :return: feature averages N x C x H x W if self.out_format == frames, EmitterSet if self.out_foramt == 'emitters'
    """

    return SpeiserPost(0.3, 0.6, 'frames').forward(x)


class CC5ChModel(ConnectedComponents):
    """Connected components on 5 channel model."""
    def __init__(self, prob_th, svalue_th=0, connectivity=2):
        super().__init__(svalue_th, connectivity)
        self.prob_th = prob_th

    @staticmethod
    def average_features(features, cluster_ix, weight):
        """
        Averages the features per cluster weighted by the probability.

        :param features: tensor (N)CHW
        :param cluster_ix: (N)HW
        :param weight: (N)HW
        :return: list of tensors of size number of clusters x features
        """

        """Add batch dimension if not already present."""
        if features.dim() == 3:
            red2hw = True  # squeeze batch dim out for return
            features = features.unsqueeze(0)
            cluster_ix = cluster_ix.unsqueeze(0)
            weight = weight.unsqueeze(0)
        else:
            red2hw = False

        batch_size = features.size(0)

        """Flatten features, weights and cluster_ix in image space"""
        feat_flat = features.view(batch_size, features.size(1), -1)
        clusix_flat = cluster_ix.view(batch_size, -1)
        w_flat = weight.view(batch_size, -1)

        feat_av = []  # list of feature average tensors
        p = []  # list of cumulative probabilites

        """Loop over the batches"""
        for i in range(batch_size):
            ccix = clusix_flat[i]  # current cluster indices in batch
            num_clusters = int(ccix.max().item())

            feat_i = torch.zeros((num_clusters, feat_flat.size(1)))
            p_i = torch.zeros(num_clusters)

            for j in range(num_clusters):
                # ix in current cluster
                ix = (ccix == j + 1)

                if ix.sum() == 0:
                    continue

                feat_i[j, :] = torch.from_numpy(
                    np.average(feat_flat[i, :, ix].numpy(), axis=1, weights=w_flat[i, ix].numpy()))
                p_i[j] = feat_flat[i, 0, ix].sum()

            feat_av.append(feat_i)
            p.append(p_i)

        if red2hw:
            return feat_av[0], p[0]
        else:
            return feat_av, p

    def forward(self, ch5_input):
        """
        Forward a batch of output of 5ch model.
        :param ch5_input: N x C=5 x H x W
        :return: emitterset
        """

        if ch5_input.dim() == 3:
            red_batch = True  # squeeze out batch dimension in the end
            ch5_input = ch5_input.unsqueeze(0)
        else:
            red_batch = False

        batch_size = ch5_input.size(0)

        """Compute connected components based on prob map."""
        p_map = ch5_input[:, 0]
        clusters = self.compute_cix(p_map)

        """Average the within cluster features"""
        feature_av, prob = self.average_features(ch5_input, clusters, p_map)  # returns list tensors of averaged feat.

        """Return list of emittersets"""
        emitter_sets = [None] * batch_size

        for i in range(batch_size):
            pi = prob[i]
            feat_i = feature_av[i]

            # get list of emitters:
            ix_above_prob_th = pi >= self.prob_th
            phot_red = feat_i[:, 1]
            xyz = torch.cat((
                feat_i[:, 2].unsqueeze(1),
                feat_i[:, 3].unsqueeze(1),
                feat_i[:, 4].unsqueeze(1)), 1)

            em = EmitterSet(xyz[ix_above_prob_th],
                            phot_red[ix_above_prob_th],
                            frame_ix=(i * torch.ones_like(phot_red[ix_above_prob_th])))

            emitter_sets[i] = em
        if red_batch:
            return emitter_sets[0]
        else:
            return emitter_sets


class CCDirectPMap(CC5ChModel):
    def __init__(self, extent, img_shape, prob_th, svalue_th=0, connectivity=2):
        """

        :param photon_threshold: minimum total value of summmed output
        :param extent:
        :param clusterer:
        :param single_value_threshold:
        :param connectivity:
        """
        super().__init__(svalue_th, connectivity)
        self.extent = extent
        self.prob_th = prob_th

        self.clusterer = label
        self.matrix_extent = None
        self.connectivity = connectivity

        self.offset2coordinate = Offset2Coordinate(extent[0], extent[1], img_shape)

        if self.connectivity == 2:
            self.kernel = np.ones((3, 3))
        elif self.connectivity == 1:
            self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def forward(self, x):
        """
        Forward a batch of frames through connected components. Must only contain one channel.

        :param x: 2D frame, or 4D batch of 1 channel frames.
        :return: (instance of emitterset)
        """

        # loop over all batch elements
        if not(x.dim() == 3 or x.dim() == 4):
            raise ValueError("Input must be C x H x W or N x C x H x W.")
        elif x.dim() == 3:
            x = x.unsquueze(0)

        """Generate a pseudo offset (with 0zeros) to use the present CC5ch model."""
        x_pseudo = torch.zeros((x.size(0), 5, x.size(2), x.size(3)))
        x_pseudo[:, 0] = x

        """Run the pseudo offsets through the Offset2Coordinate"""
        x_pseudo = self.offset2coordinate.forward(x_pseudo)

        """Run the super().forward as we are now in the same stiuation as for the 5 channel offset model."""
        return super().forward(x_pseudo)


class Offset2Coordinate:
    """Postprocesses the offset model to return a list of emitters."""
    def __init__(self, xextent, yextent, img_shape):

        off_psf = OffsetPSF(xextent=xextent,
                            yextent=yextent,
                            img_shape=img_shape)

        xv, yv = torch.meshgrid([off_psf.bin_ctr_x, off_psf.bin_ctr_y])
        self.x_mesh = xv.unsqueeze(0)
        self.y_mesh = yv.unsqueeze(0)

    def _convert_xy_offset(self, x_offset, y_offset):
        batch_size = x_offset.size(0)
        x_coord = self.x_mesh.repeat(batch_size, 1, 1).to(x_offset.device) + x_offset
        y_coord = self.y_mesh.repeat(batch_size, 1, 1).to(y_offset.device) + y_offset
        return x_coord, y_coord

    def forward(self, output):
        """
        Forwards a batch of 5ch offset model and convert the offsets to coordinates
        :param output:
        :return:
        """

        """Convert to batch if not already is one"""
        if output.dim() == 3:
            squeeze_batch_dim = True
            output = output.unsqueeze(0)
        else:
            squeeze_batch_dim = False

        """Convert the channel values to coordinates"""
        x_coord, y_coord = self._convert_xy_offset(output[:, 2], output[:, 3])

        output_converted = output.clone()
        output_converted[:, 2] = x_coord
        output_converted[:, 3] = y_coord

        if squeeze_batch_dim:
            output_converted.squeeze_(0)

        return output_converted


if __name__ == '__main__':

    speis = SpeiserPost(0.3, 0.6, 'emitters')
    speis.save('testytest.pt')
    x = torch.rand((2, 5, 64, 64))
    output = speis.forward(x)


    print("Success.")
