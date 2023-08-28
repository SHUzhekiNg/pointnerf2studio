import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import numpy as np
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs import mvs_utils, filter_utils
from pprint import pprint
from utils.visualizer import Visualizer
from utils import format as fmt
torch.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm


def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)


def nearest_view(campos, raydir, xyz):
    cam_ind = torch.zeros([0,1], device=campos.device, dtype=torch.long)
    step=10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz),i+step), None, :] - campos[None, ...] # N, M, 3
        dists_norm = torch.norm(dists, dim=-1) # N, M
        dists_dir = dists / (dists_norm[...,None]+1e-6) # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :],dim=-1)) # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1,1)], dim=0) # N, 1
    return cam_ind


def gen_points_filter_embeddings(dataset, visualizer, opt):
    print('-----------------------------------Generate Points-----------------------------------')
    opt.is_train=False
    opt.mode = 1
    model = create_model(opt)
    model.setup(opt)

    model.eval()
    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    intrinsics_full_lst = []
    confidence_filtered_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu= False #len(dataset.view_id_list) < 300

    imgs_lst, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [],[],[],[],[]
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            model.set_input(data)
            # intrinsics    1, 3, 3, 3
            points_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, c2ws, w2cs, intrinsics, near_fars  = model.gen_points()
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)
            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())
            # intrinsics_lst[0] 1, 3, 3
            intrinsics_all.append(intrinsics_lst[0] if gpu_filter else intrinsics_lst[0])
            extrinsics_all.append(extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy())
            if opt.manual_depth_view !=0:
                confidence_all.append((photometric_confidence_lst[0].cpu() if cpu2gpu else photometric_confidence_lst[0]) if gpu_filter else photometric_confidence_lst[0].cpu().numpy())
            points_mask_all.append((point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0]) if gpu_filter else point_mask_lst[0].cpu().numpy())
            imgs_lst.append(data["images"].cpu())
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            intrinsics_full_lst.append(intrinsics)
            near_fars_all.append(near_fars[0,0])
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            else:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, opt)
            # print(xyz_ref_lst[0].shape) # 224909, 3
        else:
            cam_xyz_all = [cam_xyz_all[i].reshape(-1,3)[points_mask_all[i].reshape(-1),:] for i in range(len(cam_xyz_all))]
            xyz_world_all = [np.matmul(np.concatenate([cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])], axis=-1), np.transpose(np.linalg.inv(extrinsics_all[i][0,...])))[:, :3] for i in range(len(cam_xyz_all))]
            xyz_world_all, cam_xyz_all, confidence_filtered_all = filter_utils.filter_by_masks.range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_all, opt)
            del cam_xyz_all
        # for i in range(len(xyz_world_all)):
        #     visualizer.save_neural_points(i, torch.as_tensor(xyz_world_all[i], device="cuda", dtype=torch.float32), None, data, save_ref=opt.load_points==0)
        # exit()
        # xyz_world_all = xyz_world_all.cuda()
        # confidence_filtered_all = confidence_filtered_all.cuda()
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape)
        torch.cuda.empty_cache()
        # visualizer.save_neural_points(0, xyz_world_all, None, None, save_ref=False)
        # print("vis 0")

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None:
            mask = (xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if opt.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, opt=opt)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=opt.load_points == 0)
        # print("vis 100")

        if opt.vox_res > 0:
            xyz_world_all, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all)//99999999+1),...].cuda(), opt.vox_res)
            points_vid = points_vid[sampled_pnt_idx,:]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

        xyz_world_all = [xyz_world_all[points_vid[:,0]==i, :] for i in range(len(HDWD_lst))]
        confidence_filtered_all = [confidence_filtered_all[points_vid[:,0]==i] for i in range(len(HDWD_lst))]
        cam_xyz_all = [(torch.cat([xyz_world_all[i], torch.ones_like(xyz_world_all[i][...,0:1])], dim=-1) @ extrinsics_all[i][0].t())[...,:3] for i in range(len(HDWD_lst))]
        points_embedding_all, points_color_all, points_dir_all, points_conf_all = [], [], [], []
        for i in tqdm(range(len(HDWD_lst))):
            if len(xyz_world_all[i]) > 0:
                embedding, color, dir, conf = model.query_embedding(HDWD_lst[i], torch.as_tensor(cam_xyz_all[i][None, ...], device="cuda", dtype=torch.float32), torch.as_tensor(confidence_filtered_all[i][None, :, None], device="cuda", dtype=torch.float32) if len(confidence_filtered_all) > 0 else None, imgs_lst[i].cuda(), c2ws_lst[i], w2cs_lst[i], intrinsics_full_lst[i], 0, pointdir_w=True)
                points_embedding_all.append(embedding)
                points_color_all.append(color)
                points_dir_all.append(dir)
                points_conf_all.append(conf)

        xyz_world_all = torch.cat(xyz_world_all, dim=0)
        points_embedding_all = torch.cat(points_embedding_all, dim=1)
        points_color_all = torch.cat(points_color_all, dim=1) if points_color_all[0] is not None else None
        points_dir_all = torch.cat(points_dir_all, dim=1) if points_dir_all[0] is not None else None
        points_conf_all = torch.cat(points_conf_all, dim=1) if points_conf_all[0] is not None else None

        visualizer.save_neural_points(200, xyz_world_all, points_color_all, data, save_ref=opt.load_points == 0)
        print("vis")
        model.cleanup()
        del model
    return xyz_world_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all

def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst


def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def main():
    torch.backends.cudnn.benchmark = True

    opt = TrainOptions().parse()
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    visualizer = Visualizer(opt)
    train_dataset = create_dataset(opt)
    normRw2c = train_dataset.norm_w2c[:3,:3] # torch.eye(3, device="cuda") #
    best_PSNR=0.0
    best_iter=0
    points_xyz_all=None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        # load checkpoints if exists under the given path.
        if len([n for n in glob.glob(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth") if os.path.isfile(n)]) > 0:
            resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if opt.resume_iter == "best":
                opt.resume_iter = "latest"
            resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
            if resume_iter is None:
                total_steps = 0
                visualizer.print_details("No previous checkpoints, start from scratch!!!!")
            else:
                opt.resume_iter = resume_iter
                states = torch.load(
                    os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location=cur_device)
                total_steps = states['total_steps']
                best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
                best_iter = states['best_iter'] if 'best_iter' in states else best_iter
                best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                visualizer.print_details('Continue training from {} epoch'.format(opt.resume_iter))
                visualizer.print_details(f"Iter: {total_steps}")
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                del states
            opt.mode = 2
            opt.load_points=1
            opt.resume_dir=resume_dir
            opt.resume_iter = resume_iter
            opt.is_train=True
            model = create_model(opt)
        # if no points given in the data, generate points embeddings and so on.
        elif opt.load_points < 1:
            points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all = gen_points_filter_embeddings(train_dataset, visualizer, opt)
            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train=True
            opt.mode = 2
            model = create_model(opt)
        # load points
        else:
            load_points = opt.load_points
            opt.is_train = False
            opt.mode = 1
            opt.load_points = 0
            model = create_model(opt)
            model.setup(opt)
            model.eval()
            if load_points in [1,3]:
                points_xyz_all = train_dataset.load_init_points()

            if opt.ranges[0] > -99.0:
                ranges = torch.as_tensor(opt.ranges, device=points_xyz_all.device, dtype=torch.float32)
                mask = torch.prod(
                    torch.logical_and(points_xyz_all[..., :3] >= ranges[None, :3], points_xyz_all[..., :3] <= ranges[None, 3:]),
                    dim=-1) > 0
                points_xyz_all = points_xyz_all[mask]

            if opt.vox_res > 0:
                points_xyz_all = [points_xyz_all] if not isinstance(points_xyz_all, list) else points_xyz_all
                points_xyz_holder = torch.zeros([0,3], dtype=points_xyz_all[0].dtype, device="cuda")
                for i in range(len(points_xyz_all)):
                    points_xyz = points_xyz_all[i]
                    vox_res = opt.vox_res // (1.5**i)
                    print("load points_xyz", points_xyz.shape)
                    _, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(points_xyz.cuda() if len(points_xyz) < 80000000 else points_xyz[::(len(points_xyz) // 80000000 + 1), ...].cuda(), vox_res)
                    points_xyz = points_xyz[sampled_pnt_idx, :]
                    print("after voxelize:", points_xyz.shape)
                    points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)
                points_xyz_all = points_xyz_holder

            if opt.resample_pnts > 0:
                if opt.resample_pnts == 1:
                    print("points_xyz_all",points_xyz_all.shape)
                    inds = torch.min(torch.norm(points_xyz_all, dim=-1, keepdim=True), dim=0)[1] # use the point closest to the origin
                else:
                    inds = torch.randperm(len(points_xyz_all))[:opt.resample_pnts, ...]
                points_xyz_all = points_xyz_all[inds, ...]

            campos, camdir = train_dataset.get_campos_ray()
            cam_ind = nearest_view(campos, camdir, points_xyz_all)
            unique_cam_ind = torch.unique(cam_ind)
            print("unique_cam_ind", unique_cam_ind.shape)
            points_xyz_all = [points_xyz_all[cam_ind[:,0]==unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]

            featuredim = opt.point_features_dim
            points_embedding_all = torch.zeros([1, 0, featuredim], device=unique_cam_ind.device, dtype=torch.float32)
            points_color_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_dir_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_conf_all = torch.zeros([1, 0, 1], device=unique_cam_ind.device, dtype=torch.float32)
            print("extract points embeding & colors", )
            for i in tqdm(range(len(unique_cam_ind))):
                id = unique_cam_ind[i]
                batch = train_dataset.get_item(id, full_img=True)
                HDWD = [train_dataset.height, train_dataset.width]
                c2w = batch["c2w"][0].cuda()
                w2c = torch.inverse(c2w)
                intrinsic = batch["intrinsic"].cuda()
                # cam_xyz_all 252, 4
                cam_xyz_all = (torch.cat([points_xyz_all[i], torch.ones_like(points_xyz_all[i][...,-1:])], dim=-1) @ w2c.transpose(0,1))[..., :3]
                embedding, color, dir, conf = model.query_embedding(HDWD, cam_xyz_all[None,...], None, batch['images'].cuda(), c2w[None, None,...], w2c[None, None,...], intrinsic[:, None,...], 0, pointdir_w=True)
                conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                points_color_all = torch.cat([points_color_all, color], dim=1)
                points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
            points_xyz_all=torch.cat(points_xyz_all, dim=0)
            visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
            print("vis")
            # visualizer.save_neural_points("cam", campos, None, None, None)
            # print("vis")
            # exit()

            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train = True
            opt.mode = 2
            model = create_model(opt)
        # if no checkpoints, then set the mvs-generated points to the model.
        if points_xyz_all is not None:
            model.set_points(points_xyz_all.cuda(), points_embedding_all.cuda(), points_color=points_color_all.cuda(),
                             points_dir=points_dir_all.cuda(), points_conf=points_conf_all.cuda(),
                             Rw2c=normRw2c.cuda() if opt.load_points < 1 and opt.normview != 3 else None)
            total_steps = 0
            del points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all

    model.setup(opt, train_len=len(train_dataset))
    model.train()
    data_loader = create_data_loader(opt, dataset=train_dataset)
    dataset_size = len(data_loader)
    visualizer.print_details('# training images = {}'.format(dataset_size))

    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()

    # ?
    # if total_steps > 0:
    #     for scheduler in model.schedulers:
    #         for i in range(total_steps):
    #             scheduler.step()

    if total_steps == 0:
        other_states = {
            'epoch_count': 0,
            'total_steps': total_steps,
        }
        model.save_networks(total_steps, other_states)
        visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, 0, total_steps))

    exit()


if __name__ == '__main__':
    main()
