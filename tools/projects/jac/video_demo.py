# import os
# import cv2
# import argparse

# def create_video_from_images(image_dir, output_video, fps=10):
#     """
#     Create a video from a sequence of images in the specified directory.

#     Args:
#         image_dir (str): Directory containing the images.
#         output_video (str): Path to save the output video.
#         fps (int): Frames per second for the output video.
#     """
#     # Get list of image files
#     images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
#     images.sort()  # Sort images by name to maintain the correct sequence

#     # Ensure there are images to process
#     if not images:
#         print("No images found in directory.")
#         return

#     # Read the first image to get the frame size
#     first_image_path = os.path.join(image_dir, images[0])
#     frame = cv2.imread(first_image_path)
#     height, width, layers = frame.shape

#     # Initialize the video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for .mp4 output
#     video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#     # Write each image to the video, limiting to the first 300 images
#     max_images = 300  # Set this to the maximum number of images you want to include in the video
#     for image in images[:max_images]:  
#         image_path = os.path.join(image_dir, image)
#         frame = cv2.imread(image_path)
#         video_writer.write(frame)

#     # Release the video writer
#     video_writer.release()
#     print(f"Video saved at {output_video}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Create a video from a directory of PNG images.')
#     parser.add_argument('--image_dir', type=str, required=True, help='Directory containing PNG images')
#     parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video')
#     parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video')

#     args = parser.parse_args()

#     create_video_from_images(args.image_dir, args.output_video, fps=args.fps)

import os
import cv2
import argparse

def create_video_from_images(image_dir, output_video, fps=10):
    """
    Create a video from a sequence of images in the specified directory, using only odd frames.

    Args:
        image_dir (str): Directory containing the images.
        output_video (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    # Get list of image files
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    images.sort()  # Sort images by name to maintain the correct sequence

    # Ensure there are images to process
    if not images:
        print("No images found in directory.")
        return

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for .mp4 output
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image at odd indices to the video
    odd_images = images[1::2]  # Select only images at odd indices
    for image in odd_images:  
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from a directory of PNG images using only odd frames.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing PNG images')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video')

    args = parser.parse_args()

    create_video_from_images(args.image_dir, args.output_video, fps=args.fps)

