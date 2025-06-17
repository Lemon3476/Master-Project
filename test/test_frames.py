import aPyOpenGL.agl as agl
import os

# 假设您的项目结构如仓库所示
fbx_path = os.path.join("dataset/fbx-dataset/lafan1", "test.fbx")

if os.path.exists(fbx_path):
    fbx = agl.FBX(fbx_path)
    motions = fbx.motions()
    
    print(f"在 {fbx_path} 中找到 {len(motions)} 个动画片段。")
    
    total_frames = 0
    for i, motion in enumerate(motions):
        print(f"  - 片段 {i}: {motion.num_frames} 帧")
        total_frames += motion.num_frames
    print(f"原始总帧数: {total_frames}")
else:
    print(f"未找到文件: {fbx_path}")