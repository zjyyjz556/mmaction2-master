# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import tempfile
import unittest.mock as mock
import warnings
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, IterBasedRunner
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

# TODO import eval hooks from mmcv and delete them from mmaction2
try:
    from mmcv.runner import EvalHook, DistEvalHook
    pytest.skip(
        'EvalHook and DistEvalHook are supported in MMCV',
        allow_module_level=True)
except ImportError:
    warnings.warn('DeprecationWarning: EvalHook and DistEvalHook from '
                  'mmaction2 will be deprecated. Please install mmcv through '
                  'master branch.')
    from mmaction.core import DistEvalHook, EvalHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(x=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, logger=None):
        pass


class EvalDataset(ExampleDataset):

    def evaluate(self, results, logger=None):
        acc = self.eval_result[self.index]
        output = OrderedDict(acc=acc, index=self.index, score=acc)
        self.index += 1
        return output


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    @staticmethod
    def forward(x, **kwargs):
        return x

    @staticmethod
    def train_step(data_batch, optimizer, **kwargs):
        if not isinstance(data_batch, dict):
            data_batch = dict(x=data_batch)
        return data_batch

    def val_step(self, x, optimizer, **kwargs):
        return dict(loss=self(x))


def _build_epoch_runner():

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=get_logger('demo'))
    return runner


def _build_iter_runner():

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = IterBasedRunner(
        model=model, work_dir=tmp_dir, logger=get_logger('demo'))
    return runner


def test_eval_hook():
    with pytest.raises(AssertionError):
        # `save_best` should be a str
        test_dataset = Model()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best=True)

    with pytest.raises(TypeError):
        # dataloader must be a pytorch DataLoader
        test_dataset = Model()
        data_loader = [DataLoader(test_dataset)]
        EvalHook(data_loader)

    with pytest.raises(ValueError):
        # save_best must be valid when rule_map is None
        test_dataset = ExampleDataset()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best='unsupport')

    with pytest.raises(KeyError):
        # rule must be in keys of rule_map
        test_dataset = Model()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best='auto', rule='unsupport')

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset)
    model = Model()
    data_loader = DataLoader(test_dataset)
    eval_hook = EvalHook(data_loader, save_best=None)

    with tempfile.TemporaryDirectory() as tmpdir:

        # total_epochs = 1
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with(
            test_dataset, [torch.tensor([1])], logger=runner.logger)
        assert runner.meta is None or 'best_score' not in runner.meta[
            'hook_msgs']
        assert runner.meta is None or 'best_ckpt' not in runner.meta[
            'hook_msgs']

    # when `save_best` is set to 'auto', first metric will be used.
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, interval=1, save_best='auto')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best acc and corresponding epoch
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, interval=1, save_best='acc')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best score and corresponding epoch
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(
        data_loader, interval=1, save_best='score', rule='greater')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_score_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best score using less compare func
    # and indicate corresponding epoch
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, save_best='acc', rule='less')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_6.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == -3

    # Test the EvalHook when resume happend
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, save_best='acc')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_2.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 4

        resume_from = osp.join(tmpdir, 'latest.pth')
        loader = DataLoader(ExampleDataset())
        eval_hook = EvalHook(data_loader, save_best='acc')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.resume(resume_from)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(ckpt_path)
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7


@patch('mmaction.apis.single_gpu_test', MagicMock)
@patch('mmaction.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', [EvalHook, DistEvalHook])
@pytest.mark.parametrize('_build_demo_runner,by_epoch',
                         [(_build_epoch_runner, True),
                          (_build_iter_runner, False)])
def test_start_param(EvalHookParam, _build_demo_runner, by_epoch):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, interval=1, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=1, interval=1, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, interval=2, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=1, interval=2, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/negative, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=0, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    runner = _build_demo_runner()
    with pytest.warns(UserWarning):
        evalhook = EvalHookParam(
            dataloader, start=-2, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=1, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    if by_epoch:
        runner._epoch = 2
    else:
        runner._iter = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=2, by_epoch=by_epoch, save_best=None)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    if by_epoch:
        runner._epoch = 1
    else:
        runner._iter = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3

    shutil.rmtree(runner.work_dir)
