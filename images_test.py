import cv2
import matplotlib.pyplot as plt

# Load image (replace with your image path)
image = cv2.imread("road.jpg")

# Print shape
print("Image shape:", image.shape)

# Convert BGR to RGB (important for correct colors)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray_image.shape)

# Show grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
