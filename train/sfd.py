font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
# set the rectangLe background to white
rectang1e_bgr : (255, 255, 255)

# make a bLack image
img = np.zeros((500, 500))

# set some text
text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale= font_scale, thickness=1)[0]
# set the text start position
text_offset_x : 10
text_offset_y = img.shape[0] - 25
# make the coords of the box with a smaLL padding of two pixeLs
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y — text_height — 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctLy
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():  
    raise IOError("Cannot open webcam")