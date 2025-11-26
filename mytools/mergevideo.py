from moviepy.editor import VideoFileClip, clips_array

# 读取视频
video1 = VideoFileClip("/mnt/data/lyl/codes/RGBAvatar/output/INSTA/nf_03/reproduction/render_image/GT_350.avi")
video2 = VideoFileClip("/mnt/data/lyl/codes/RGBAvatar/output/INSTA/nf_03/reproduction/render_image/video.avi")
video3 = VideoFileClip("/mnt/data/lyl/codes/RGBAvatar/output/INSTA/nf_03/bbw_bbw500/render_image/350.avi")


# 高度自动对齐
final_video = clips_array([
    [video1, video2, video3]  # 横向拼接
])

# 导出视频
final_video.write_videofile(
    "/mnt/data/lyl/codes/RGBAvatar/output/INSTA/nf_03/gt_paper_ours500.mp4",
    codec="libx264",
    audio_codec="aac"
)
