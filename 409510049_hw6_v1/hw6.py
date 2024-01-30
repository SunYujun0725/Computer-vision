import cv2
import numpy as np

def rotate_homogeneous(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Define rotation matrix in homogeneous coordinates
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])

    # Define the center of rotation
    center_x, center_y = width / 2, height / 2
    # Define the bottom-left corner of rotation
    #center_x, center_y = 0, height-1

    # Iterate through each pixel in the original image
    warped_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            # Apply inverse transformation to find corresponding pixel in original image
            new_coords = np.dot(rotation_matrix, np.array([x - center_x, y - center_y, 1]))
            new_x, new_y = int(new_coords[0] + center_x), int(new_coords[1] + center_y)

            # Check if the new coordinates are within the bounds of the original image
            if 0 <= new_x < width and 0 <= new_y < height:
                # Assign pixel value from original image to corresponding location in the rotated image
                warped_image[new_y, new_x] = image[y, x]

    return warped_image

def main():
    # Load the image
    original_image = cv2.imread("resized.jpg")

    # Rotate the image by 45 degrees using homogeneous coordinates
    # 45 / -45
    rotated_image = rotate_homogeneous(original_image, 45)

    # Display the rotated images
    cv2.imwrite("./rotated_image.jpg", rotated_image)


if __name__ == '__main__':
    main()