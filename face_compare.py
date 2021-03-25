import face_recognition


def face_compare():
    known_images_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    unknown_image_path = "unknown.jpg"

    known_encodings = []
    for known_image_path in known_images_paths:
        encoding = face_recognition.face_encodings(face_recognition.load_image_file(known_image_path))
        if len(encoding) > 0:
            known_encodings.append(encoding[0])

    unknown_encoding = face_recognition.face_encodings(face_recognition.load_image_file(unknown_image_path))[0]

    results = face_recognition.compare_faces(known_encodings, unknown_encoding)
    return results


if __name__ == "__main__":
    print(face_compare())
