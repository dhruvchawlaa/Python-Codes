from mtcnn import MTCNN
import cv2
# Create a detector instance
detector = MTCNN()

# Load an image
image = cv2.imread('smile_dataset/pos/file0001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image
result = detector.detect_faces(image)

# Display the result
print(result)