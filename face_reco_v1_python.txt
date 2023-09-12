import face_recognition
import cv2

# Paths to the images
path_to_shlomi = "C:\\Users\\Yuval Kahan\\Downloads\\face recognation\\toberecognaized.PNG"
path_to_two_people = "C:\\Users\\Yuval Kahan\\Downloads\\face recognation\\today_intermittent_fasting_1_month.png"

# Load the image of Shlomi that you want to recognize
image_of_shlomi = face_recognition.load_image_file(path_to_shlomi)

# Encode Shlomi's face
shlomi_face_encoding = face_recognition.face_encodings(image_of_shlomi)[0]

# Create an array of known face encodings and their names
known_face_encodings = [shlomi_face_encoding]
known_face_names = ["Shlomi"]

# Load the image with two people that you want to check
unknown_image = face_recognition.load_image_file(path_to_two_people)

# Find all face locations and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to an OpenCV format
image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match is found, use the known face name
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a box around the face and label it
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the image
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
