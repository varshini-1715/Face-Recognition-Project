import os, zipfile, gdown, cv2, numpy as np, matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

file_id = "1jGiDt4VM9flUtqtBsc1sAnNwE4ekTo-x"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "DL_PROJECT.zip", quiet=False)

with zipfile.ZipFile("DL_PROJECT.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

train_path = "DL_PROJECT/train"
test_path = "DL_PROJECT/test"

def load_image_correctly(path):
    img = Image.open(path)
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))

embeddings_db = []
labels_db = []

for person in os.listdir(train_path):
    person_clean = person.strip()
    person_path = os.path.join(train_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = load_image_correctly(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            continue

        embeddings_db.append(faces[0].embedding)
        labels_db.append(person_clean)

embeddings_db = np.array(embeddings_db)

def recognize_face(embedding):
    similarity = cosine_similarity([embedding], embeddings_db)[0]
    best_index = np.argmax(similarity)
    confidence = similarity[best_index]

    if confidence > 0.6:
        return labels_db[best_index], confidence
    return "Unknown", confidence

# ===================== PROCESS ALL TEST IMAGES =====================

for person in sorted(os.listdir(test_path)):

    person_clean = person.strip()
    person_path = os.path.join(test_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in sorted(os.listdir(person_path)):

        img_path = os.path.join(person_path, img_name)
        image = load_image_correctly(img_path)

        if image is None:
            continue

        faces = app.get(image)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.embedding
            pred_label, conf = recognize_face(embedding)

            # 🔥 BIG TEXT + THICK BOX (CLEAR VISIBILITY)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 4)

            cv2.putText(image,
                        f"{pred_label} ({conf:.2f})",
                        (x1, y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,            # BIG TEXT
                        (255,0,0),
                        4)              # THICK TEXT

        print(f"Actual: {person_clean} | Predicted: {pred_label}")

        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Actual: {person_clean} | Pred: {pred_label}")
        plt.axis("off")
        plt.show()