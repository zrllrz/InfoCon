import os, argparse
import torch
import clip
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PegInsertionSide-v0', help="Task (env-id) in ManiSkill2.")
    return parser.parse_args()


if __name__ == "__main__":

    KEY_PC = ["The robot is grasping the red cube"]
    KEY_SC = ["The robot has just grasped the red cube.",
              "The red cube is on top of the green cube."]
    KEY_TF = ["The robot is concatenate with the faucet."]
    KEY_PIS = ["The robot is grasping the peg",
               "The peg is align with the hole"]

    args = parse_args()
    frames_set_path = '../CoTPC-main/data/' + args.task + '/frames/'

    device = "cuda"
    print('loading model')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device)
    print('loading model done')

    if args.task == 'PegInsertionSide-v0':
        args_keys = KEY_PIS
    elif args.task == 'StackCube-v0':
        args_keys = KEY_SC
    elif args.task == 'PickCube-v0':
        args_keys = KEY_PC
    elif args.task == 'TurnFaucet-v0':
        args_keys = KEY_TF
    else:
        print('unimplement task')
        assert False

    key_state_description_token = clip.tokenize(args_keys).to(device)
    key_state_text_features = clip_model.encode_text(key_state_description_token)

    with open('../CoTPC-main/data/' + args.task + '/keys-clip.txt', 'w') as fk:
        with torch.no_grad():
            i_traj = 0
            while os.path.exists(frames_set_path + f'{i_traj}'):
                i_frame = 0
                frame_list = []
                while os.path.exists(frames_set_path + f'{i_traj}/{i_frame}.jpg'):
                    fr = Image.open(frames_set_path + f'{i_traj}/{i_frame}.jpg')
                    fr = clip_preprocess(fr).unsqueeze(0).to(device)
                    frame_list.append(fr)

                    i_frame += 1

                frames = torch.cat(frame_list, dim=0)
                frames_feature = clip_model.encode_image(frames)

                score = torch.cosine_similarity(
                    x1=key_state_text_features.unsqueeze(1).repeat(1, i_frame, 1),
                    x2=frames_feature.unsqueeze(0).repeat(len(args_keys), 1, 1),
                    dim=-1
                )

                k_max = torch.max(score, dim=-1, keepdim=True)[0]
                k_min = torch.min(score, dim=-1, keepdim=True)[0]

                idx_mat = torch.arange(i_frame, device=device).unsqueeze(0).repeat(len(args_keys), 1)

                idx_mat = torch.where(torch.ge(score, (k_max + k_min) * 0.5), idx_mat, 10000)

                key_states_idx = torch.min(idx_mat, dim=-1)[0]
                key_states_idx = torch.where(torch.eq(key_states_idx, 10000), -1, key_states_idx)

                key_state_str = ''
                for idx in key_states_idx:
                    key_state_str += (str(idx.item()) + ',')
                key_state_str += (str(i_frame - 1) + ',')
                print(f"key states {i_traj}:", key_state_str)
                fk.write(key_state_str + '\n')
                i_traj += 1

