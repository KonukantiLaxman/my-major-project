from moviepy.editor import TextClip, CompositeVideoClip

# Create a text clip
text = "Get fucking real dude."
text_clip = TextClip(text, fontsize=70, color='white', bg_color='black', size=(1280, 720))
text_clip = text_clip.set_duration(5)  # 5-second duration

# Export the video as MP4
text_clip.write_videofile("text_video.mp4", fps=24)
