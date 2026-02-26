import kagglehub

# Download latest version
path = kagglehub.dataset_download("toshihikochen/doclaynet-v1-2-yolo",output_dir="/run/media/akshajtiwari/New Volume/Doclaynet_yolo")

print("Path to dataset files:", path)