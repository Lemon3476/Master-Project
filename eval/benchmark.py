import sys
sys.path.append(".")

from tqdm import tqdm
import argparse

import torch

from utils import eval, utils
from utils.eval import _segment_net_transition
from utils.dataset import MotionDataset
from model.segment_net import SegmentNet
from model.twostage import ContextTransformer

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument must be a list")
    return v

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--interp", type=lambda s: s.lower() in ['true', '1'])
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--ts_configs", type=str, nargs="+", default=[]) # odd: context, even: detail
    parser.add_argument("--rmi_configs", type=str, nargs="+", default=[])
    parser.add_argument("--ours_configs", type=str, nargs="+", default=[]) # odd: keyframe, even: refine
    parser.add_argument("--traj_edit", type=str, default=None)
    # Removed random_kf_replace and reverse_kf_replace arguments
    parser.add_argument("--intra_batch_replace", action="store_true", help="Replace middle keyframe with keyframe from other sequences in same batch (0->1, 1->2, ..., 63->0)")
    parser.add_argument("--replace_target", action="store_true", help="Replace target frame with frame from other sequences in same batch and refine")
    parser.add_argument("--multi_ib", action="store_true", help="Use multi-stage in-betweening generation with frame modification")
    # SegmentNet specific arguments
    parser.add_argument("--use_segment", action="store_true", help="Use SegmentNet for motion generation")
    parser.add_argument("--only_segment", action="store_true", help="Use only SegmentNet without baseline methods")
    parser.add_argument("--kf_config", type=str, default="keyframe.yaml", help="KeyframeNet config file (for SegmentNet)")
    parser.add_argument("--seg_config", type=str, default="segment.yaml", help="SegmentNet config file")
    parser.add_argument("--kf_sampling", type=str, default="score", help="Keyframe sampling method (score, topk, threshold, random, uniform)")
    parser.add_argument("--kf_param", type=float, default=None, help="Parameter for keyframe sampling")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Parse keyframe sampling arguments
    if args.kf_param is not None:
        kf_sampling = [args.kf_sampling, args.kf_param]
    else:
        kf_sampling = [args.kf_sampling]
    
    # evaluator
    evaluator = eval.Evaluator(args)
    transitions = [15, 30, 60, 90]
    # transitions = [15, ]
    results = {
        "tags": None,
        "l2p": [],
        "l2q": [],
        "npss": [],
        "foot skate": [],
    }
    
    # Initialize SegmentNet if needed
    segment_model = None
    keyframe_model = None
    dataset = None
    
    # Initialize tags list if it's None
    if results["tags"] is None:
        results["tags"] = []
    
    if args.use_segment:
        print("Loading SegmentNet model...")
        # Load dataset for statistics
        dataset = MotionDataset(train=False, config=evaluator.config)
        
        # Statistics for normalization
        mean, std = dataset.motion_statistics()
        mean, std = mean.to(device), std.to(device)
        
        traj_mean, traj_std = dataset.traj_statistics()
        traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
        
        # Load KeyframeNet model
        kf_config = utils.load_config(f"config/{args.dataset}/{args.kf_config}")
        keyframe_model = ContextTransformer(kf_config, dataset).to(device)
        utils.load_model(keyframe_model, kf_config)
        keyframe_model.eval()
        
        # Load SegmentNet model
        seg_config = utils.load_config(f"config/{args.dataset}/{args.seg_config}")
        segment_model = SegmentNet(seg_config, dataset).to(device)
        utils.load_model(segment_model, seg_config)
        segment_model.eval()
        
        print(f"SegmentNet loaded successfully. Using keyframe sampling method: {args.kf_sampling}")
        
        # Add SegmentNet to the list of methods to evaluate
        if "SegmentNet" not in results["tags"]:  # Prevent duplicate entries
            results["tags"].append("SegmentNet")
        print(f"Added SegmentNet to methods. Current methods: {results['tags']}")
    for trans in tqdm(transitions):
        num_frames = evaluator.config.context_frames + trans + 1
        motion_list, contact_list = [], []
        traj_list = []
        tags, skeleton = None, None
        
        # Get results from baseline methods
        baseline_results = []
        for idx, res in enumerate(evaluator.eval(num_frames, traj_option=args.traj_edit)):
            baseline_results.append(res)
            if idx == 0:
                skeleton = res["skeleton"]
                # Add baseline method tags
                baseline_tags = res["tags"][1:]  # Skip GT
                for tag in baseline_tags:
                    if tag not in results["tags"]:
                        results["tags"].append(tag)
                print(f"Added baseline methods. Updated tags: {results['tags']}")
        
        # Add baseline results to the evaluation lists
        for res in baseline_results:
            motion_list.append(res["motions"])
            contact_list.append(res["contacts"])
            traj_list.append(res["trajs"])
        
        # Add SegmentNet results if enabled
        if args.use_segment and segment_model is not None and keyframe_model is not None:
            # Get the GT data from the first batch
            GT_motion = baseline_results[0]["motions"][0]  # GT motion
            GT_contact = baseline_results[0]["contacts"][0]  # GT contact
            GT_traj = baseline_results[0]["trajs"][0]  # GT trajectory
            
            # Get phase from dataset if needed
            GT_phase = None
            if seg_config.use_phase:
                batch_data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=GT_motion.shape[0])))
                GT_phase = batch_data["phase"].to(device)
                if GT_phase.shape[1] > num_frames:
                    GT_phase = GT_phase[:, :num_frames]
            
            # Get score from dataset if needed
            GT_score = None
            if seg_config.use_score:
                batch_data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=GT_motion.shape[0])))
                GT_score = batch_data["score"].to(device)
                if GT_score.shape[1] > num_frames:
                    GT_score = GT_score[:, :num_frames]
            
            # Generate motion using SegmentNet
            with torch.no_grad():
                segment_result = _segment_net_transition(
                    seg_config,
                    keyframe_model,
                    segment_model,
                    GT_motion,
                    mean,
                    std,
                    GT_contact,
                    GT_phase,
                    GT_traj,
                    GT_score,
                    traj_mean,
                    traj_std,
                    kf_sampling
                )
                print(f"SegmentNet result obtained. Keys: {segment_result.keys()}")
            
            # Create lists in the same format as other methods
            segment_motions = [GT_motion]  # First element is GT
            segment_motion = segment_result["motion"]
            print(f"SegmentNet motion shape: {segment_motion.shape}, GT shape: {GT_motion.shape}")
            
            # Make sure the shape matches GT
            if segment_motion.shape[1] != GT_motion.shape[1]:
                print(f"WARNING: Shape mismatch between SegmentNet result ({segment_motion.shape}) and GT ({GT_motion.shape})")
                # Pad or trim to match GT frame count
                if segment_motion.shape[1] < GT_motion.shape[1]:
                    padding = torch.zeros((segment_motion.shape[0], GT_motion.shape[1] - segment_motion.shape[1], segment_motion.shape[2]), device=segment_motion.device)
                    segment_motion = torch.cat([segment_motion, padding], dim=1)
                    print(f"Padded SegmentNet motion to {segment_motion.shape}")
                else:
                    segment_motion = segment_motion[:, :GT_motion.shape[1]]
                    print(f"Trimmed SegmentNet motion to {segment_motion.shape}")
            
            segment_motions.append(segment_motion)  # Second element is SegmentNet result
            
            segment_contacts = [GT_contact]  # First element is GT
            segment_contact = segment_result.get("contact", GT_contact)
            
            # Make sure contact shape matches GT
            if segment_contact.shape[1] != GT_contact.shape[1]:
                print(f"WARNING: Contact shape mismatch between SegmentNet result ({segment_contact.shape}) and GT ({GT_contact.shape})")
                # Pad or trim to match GT frame count
                if segment_contact.shape[1] < GT_contact.shape[1]:
                    padding = torch.zeros((segment_contact.shape[0], GT_contact.shape[1] - segment_contact.shape[1], segment_contact.shape[2]), device=segment_contact.device)
                    segment_contact = torch.cat([segment_contact, padding], dim=1)
                    print(f"Padded SegmentNet contact to {segment_contact.shape}")
                else:
                    segment_contact = segment_contact[:, :GT_contact.shape[1]]
                    print(f"Trimmed SegmentNet contact to {segment_contact.shape}")
            
            segment_contacts.append(segment_contact)  # Second element is SegmentNet result
            
            segment_trajs = [GT_traj]  # First element is GT
            segment_trajs.append(GT_traj)  # Use GT trajectory
            
            # Add SegmentNet results to the evaluation lists
            motion_list.append(segment_motions)
            contact_list.append(segment_contacts)
            traj_list.append(segment_trajs)
            print(f"Added SegmentNet results. Motion shape: {segment_motions[1].shape}")
        
        # Concat all results (0: GT, 1~: others)
        # First concatenate all GT motions
        gt_motions = torch.cat([batch[0] for batch in motion_list], dim=0)
        gt_contacts = torch.cat([batch[0] for batch in contact_list], dim=0)
        gt_trajs = torch.cat([batch[0] for batch in traj_list], dim=0)
        
        # Then concatenate all method results
        method_motions = []
        method_contacts = []
        method_trajs = []
        
        # Get the number of methods (excluding GT)
        num_methods = len(results["tags"])
        print(f"Number of methods to evaluate: {num_methods}, Tags: {results['tags']}")
        
        # If we only have SegmentNet (no baseline methods), make sure we handle it correctly
        only_segment = (num_methods == 1 and results["tags"][0] == "SegmentNet")
        
        for method_idx in range(num_methods):
            # Collect this method's results across all batches
            method_motion_list = []
            method_contact_list = []
            method_traj_list = []
            
            for batch_idx, batch in enumerate(motion_list):
                # For baseline methods
                if batch_idx < len(baseline_results):
                    if method_idx + 1 < len(batch):  # +1 to skip GT
                        method_motion_list.append(batch[method_idx + 1])
                        method_contact_list.append(contact_list[batch_idx][method_idx + 1])
                        method_traj_list.append(traj_list[batch_idx][method_idx + 1])
                # For SegmentNet (which is the last method unless it's the only method)
                elif args.use_segment and ((method_idx == num_methods - 1 and not only_segment) or 
                                          (only_segment and results["tags"][method_idx] == "SegmentNet")) and len(batch) > 1:
                    method_motion_list.append(batch[1])  # SegmentNet result
                    method_contact_list.append(contact_list[batch_idx][1])
                    method_traj_list.append(traj_list[batch_idx][1])
                    print(f"Processing SegmentNet result for batch {batch_idx}, shape: {batch[1].shape}")
            
            # Concatenate this method's results
            if method_motion_list:
                # 首先，将所有批次拼接成一个大张量
                concatenated_motions = torch.cat(method_motion_list, dim=0)
                
                # 然后，打印这个大张量的长度（即样本总数）
                print(f"Method {method_idx} ({results['tags'][method_idx]}): {len(concatenated_motions)} motion samples")
                
                method_motions.append(concatenated_motions)
                try:
                    method_motions.append(torch.cat(method_motion_list, dim=0))
                    method_contacts.append(torch.cat(method_contact_list, dim=0))
                    method_trajs.append(torch.cat(method_traj_list, dim=0))
                except Exception as e:
                    print(f"Error concatenating results for method {method_idx}: {e}")
                    # Print shape information for debugging
                    for i, m in enumerate(method_motion_list):
                        print(f"  Motion {i} shape: {m.shape}")
                    for i, c in enumerate(method_contact_list):
                        print(f"  Contact {i} shape: {c.shape}")
                    # Try to use only the first batch as a fallback
                    if len(method_motion_list) > 0:
                        method_motions.append(method_motion_list[0])
                        method_contacts.append(method_contact_list[0])
                        method_trajs.append(method_traj_list[0])
        
        # Prepare final lists for evaluation
        final_motion_list = [gt_motions] + method_motions
        final_contact_list = [gt_contacts] + method_contacts
        final_traj_list = [gt_trajs] + method_trajs

        # L2P
        l2p_list = []
        for motion in final_motion_list[1:]:
            l2p = eval.l2p(final_motion_list[0], motion, skeleton, evaluator.l2p_mean, evaluator.l2p_std, evaluator.config.context_frames)
            l2p_list.append(l2p)

        # L2Q
        l2q_list = []
        for motion in final_motion_list[1:]:
            l2q = eval.l2q(final_motion_list[0], motion, evaluator.config.context_frames)
            l2q_list.append(l2q)

        # NPSS
        npss_list = []
        for motion in final_motion_list[1:]:
            npss = eval.npss(final_motion_list[0], motion, evaluator.config.context_frames)
            npss_list.append(npss)
        
        # Foot skate
        fs_list = []
        for motion, contact in zip(final_motion_list[1:], final_contact_list[1:]):
            fs = eval.foot_skate(motion, contact, skeleton, evaluator.contact_idx, ctx_frames=evaluator.config.context_frames)
            fs_list.append(fs)
        
        # Optional: traj position error
        if args.traj_edit is not None:
            pos_err_list = []
            for motion, traj in zip(final_motion_list[1:], final_traj_list[1:]):
                pos_err = eval.traj_pos_error(traj, motion, evaluator.config.context_frames)
                pos_err_list.append(pos_err)

        results["l2p"].append(l2p_list)
        results["l2q"].append(l2q_list)
        results["npss"].append(npss_list)
        results["foot skate"].append(fs_list)

        if args.traj_edit is not None:
            if "traj" not in results:
                results["traj"] = []
            results["traj"].append(pos_err_list)

    # save in text for latex table
    def get_row(method, metric, idx):
        row = str(method)
        for i in range(len(transitions)):
            if metric in ["l2p", "l2q"]:
                row += f" & {results[metric][i][idx]:.2f}"
            else:
                row += f" & {results[metric][i][idx] * 10:.2f}"
        return row + " \\\\\n"

    # save in text for latex table
    # Add segment suffix if SegmentNet is used
    output_filename = f"eval/benchmark-{args.dataset}{'-segment' if args.use_segment else ''}.txt"
    with open(output_filename, "w") as f:
        metric_list = ["l2p", "l2q", "npss", "foot skate"]
        if args.traj_edit is not None:
            metric_list.append("traj")
        for metric in metric_list:
            f.write(f"{metric.upper() if metric != 'foot skate' else 'Foot skate'}" + "\n")

            row = f"$t_\\mathrm{{trans}}$"
            for t in transitions:
                row += f" & {t}"
            f.write(f"{row} \\\\ \\midrule\n")

            for i in range(len(results["tags"])):
                f.write(get_row(results["tags"][i], metric, i))
            f.write("\n")
    
    # Print a summary of the results
    print(f"\nResults saved to {output_filename}")
    print("\nResults summary:")
    for metric in ["l2p", "l2q", "npss", "foot skate"]:
        print(f"\n{metric.upper() if metric != 'foot skate' else 'Foot skate'}")
        
        # Header with transition lengths
        header = "Method"
        for t in transitions:
            header += f"\t{t}"
        print(header)
        
        # Results for each method
        for i, tag in enumerate(results["tags"]):
            row = f"{tag}"
            for j in range(len(transitions)):
                if metric in ["l2p", "l2q"]:
                    row += f"\t{results[metric][j][i]:.2f}"
                else:
                    row += f"\t{results[metric][j][i] * 10:.2f}"
            print(row)