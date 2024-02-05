# smooth_pose.py

import torch
import numpy as np

# from ..models.head.smpl_head import SMPL
from ..models.head.smplx_local import SMPLX
from ..core import config
from ..core.config import SMPL_MODEL_DIR
from .one_euro_filter import OneEuroFilter


def smooth_pose(pred_pose, pred_betas, min_cutoff=0.004, beta=0.7):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0].cpu().numpy()),
        pred_pose[0].cpu().numpy(),
        min_cutoff=min_cutoff,
        beta=beta,
    )

    # smpl = SMPL(model_path=SMPL_MODEL_DIR)

    pred_pose_hat = np.zeros_like(pred_pose.cpu().numpy()) # np.zeros_like(pred_pose)

    # initialize
    # pred_pose_hat[0] = pred_pose[0]
    pred_pose_hat[0] = pred_pose[0].detach().cpu().numpy()

    pred_verts_hat = []
    pred_joints3d_hat = []

    smpl_output = smplx(
        betas=torch.from_numpy(pred_betas[0].detach().cpu().numpy()).unsqueeze(0), # betas=torch.from_numpy(pred_betas[0]).unsqueeze(0),
        body_pose=torch.from_numpy(pred_pose[0, 1:].detach().cpu().numpy()).unsqueeze(0),
        global_orient=torch.from_numpy(pred_pose[0, 0:1].detach().cpu().numpy()).unsqueeze(0),
        pose2rot=False,
    )
    pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
    print(f"len pred_verts_hat1 : {len(pred_verts_hat[0][0])}")
    pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())
    # print(f"pred_pose[0][1:] = {pred_pose[0][1:]}")

    pred_verts_hat_list = []
    pred_joints3d_hat_list = []
    for idx, pose in enumerate(pred_pose[0][1:]): # pred_pose[1:] -> pred_pose[0][1:]로 수정
        print("for문 돌아감")
        idx += 1

        if idx >= len(pred_pose_hat[0]):  # 수정: pred_pose_hat[0]의 길이로 체크
            idx = len(pred_pose_hat[0]) - 1 

        t = np.ones_like(pose.detach().cpu().numpy()) * idx
        pose = one_euro_filter(t, torch.tensor(pose))  # pose -> torch.tensor(pose)
        # print(f"idx : {idx}")
        print(f"pred_pose_hat : {pred_pose_hat}") 
        pred_pose_hat[0][idx] = pose[0]
        # pred_pose_hat[idx] = pose

        betas = torch.from_numpy(pred_betas[idx].detach().cpu().numpy()).unsqueeze(0)

        smpl_output = smplx(
            betas=betas, # torch.from_numpy(pred_betas[idx]).unsqueeze(0),
            body_pose = torch.from_numpy(pred_pose_hat[idx, 1:]).unsqueeze(0).detach().cpu(),
            global_orient = torch.from_numpy(pred_pose_hat[idx, 0:1]).unsqueeze(0).detach().cpu(),
            pose2rot=False,
        )
        # pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        # print(f"len pred_verts_hat2 : {len(pred_verts_hat[0][0])}")
        # pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())
        pred_verts_hat_list.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat_list.append(smpl_output.joints.detach().cpu().numpy())

        # return값 변수 지정하여 변경
        pred_verts_hat = np.vstack(pred_verts_hat_list)
        pred_joints3d_hat = np.vstack(pred_joints3d_hat_list)
    
    return pred_verts_hat, pred_pose_hat, pred_joints3d_hat