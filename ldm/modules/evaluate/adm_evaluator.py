import argparse
import io
import os 
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
import yaml

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"

REQUIREMENTS = f"This script has the following requirements: \n" \
               'tensorflow-gpu>=2.0' + "\n" + 'scipy' + "\n" + "requests" + "\n" + "tqdm"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_batch', help="path to reference batch npz file")
    parser.add_argument("--sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()

    config = tf.ConfigProto(allow_soft_placement=True) # allows DecodeJpeg to run on CPU in Inception graph
    config.gpu_optios.allow_growth = True 
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)

    print("Computing evaluations...")
    is_ = evaluator.compute_inception_score(sample_acts[0])
    print("Inception Score:", is_)
    fid = sample_stats.frechet_distance(ref_stats)
    print("FID:", fid)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    print("sFID:", sfid)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print("Precision:", prec)
    print("Recall:", recall)

    savepath = '/'.join(args.sample_batch.split('/')[:-1])
    results_file = os.path.join(savepath,'evaluation_metrics.yaml')
    print(f'Saving evaluation results to "{results_file}"')

    results = {
        'IS': is_,
        'FID': fid,
        'sFID': sfid,
        'Precision:':prec,
        'Recall': recall
    }

    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)


class InvalidFIDException(Exception):
    pass

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    
    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_1d(sigma1)
        sigma2 = np.atleast_1d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2 

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        
        tg_covmean = np.trace(convmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class Evaluater:
    def __init__(self, 
                 session, 
                 batch_size=64,
                 softmax_batch_size=512):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)
    
    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))
    
    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))
    
    


