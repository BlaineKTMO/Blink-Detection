import cv2
import blink_detector
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.nn import functional as F
import time

class ViTForImageClassificationWithSoftmax(ViTForImageClassification):
    def forward(self, pixel_values, labels=None):
        outputs = super().forward(pixel_values, labels)
        logits = outputs.logits
        softmax_logits = F.softmax(logits, dim=-1)
        return softmax_logits

fps = 1
delay = 1 / fps

def main():
    model_path="google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=2)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    model.load_state_dict(torch.load('./model_new.pth'))
    model.eval()
    print(model)

    cap = cv2.VideoCapture(2)

    start_time =time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (105, 105))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        features = feature_extractor(images=frame, return_tensors="pt")
        features = {k: v.squeeze(0) for k, v in features.items()}
        output = model(features["pixel_values"].unsqueeze(0)).logits
        prediction = output.argmax(dim=1)
        print(output)

        if output[0][1] > 0.15:
            count += 1
            print("Blink Detected")

        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        time.sleep(max(0, delay-elapsed_time))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()