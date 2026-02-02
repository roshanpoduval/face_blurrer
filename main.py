import os
import argparse

import cv2
import mediapipe as mp
import numpy as np


class FaceBlurrer:
    def __init__(self, blur_amount=30, blur_type='oval', feather=10, scale=1.0, oval_stretch=0.85, persistence=5):
        self.blur_amount = blur_amount
        self.blur_type = blur_type
        self.feather = feather
        self.scale = scale
        self.oval_stretch = oval_stretch
        self.persistence = persistence
        self.last_bbox = None
        self.persistence_count = 0

    def process_img(self, img, face_detection):
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        if out.detections:
            self.persistence_count = self.persistence
            # For simplicity, handle one face
            detection = out.detections[0]
            location_data = detection.location_data
            self.last_bbox = location_data.relative_bounding_box
        elif self.last_bbox and self.persistence_count > 0:
            self.persistence_count -= 1
        else:
            self.last_bbox = None

        if self.last_bbox:
            bbox = self.last_bbox
            x1_orig, y1_orig, w_orig, h_orig = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1_orig = int(x1_orig * W)
            y1_orig = int(y1_orig * H)
            w_orig = int(w_orig * W)
            h_orig = int(h_orig * H)
            
            # Calculate new scaled bounding box
            x_center = x1_orig + w_orig // 2
            y_center = y1_orig + h_orig // 2

            w_new = int(w_orig * self.scale)
            h_new = int(h_orig * self.scale)

            x1 = int(x_center - w_new // 2)
            y1 = int(y_center - h_new // 2)
            x2 = x1 + w_new
            y2 = y1 + h_new

            # Clamp coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            w = x2 - x1
            h = y2 - y1
            
            if w > 0 and h > 0:
                if self.blur_type == 'rectangle':
                    img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (self.blur_amount, self.blur_amount))
                
                elif self.blur_type == 'oval':
                    face_region = img[y1:y2, x1:x2]
                    blurred_face = cv2.blur(face_region, (self.blur_amount, self.blur_amount))
                    mask = np.zeros((h, w), dtype=np.uint8)
                    axes = (int(w // 2 * self.oval_stretch), h // 2)
                    cv2.ellipse(mask, (w // 2, h // 2), axes, 0, 0, 360, (255), -1)
                    
                    sigma_y = self.feather * (h / w) if w != 0 else self.feather
                    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, self.feather), sigmaY=max(1, int(sigma_y)))
                    mask = mask / 255.0
                    
                    result_face = (blurred_face * mask[:, :, np.newaxis]).astype(np.uint8) + \
                                  (face_region * (1 - mask[:, :, np.newaxis])).astype(np.uint8)
                    
                    img[y1:y2, x1:x2] = result_face
        return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args.add_argument("--blur", type=int, default=150, help="Amount of blur to apply (kernel size)")
args.add_argument("--blur-type", type=str, default='oval', choices=['oval', 'rectangle'], help="Type of blur to apply")
args.add_argument("--feather", type=int, default=6, help="Amount of feathering for oval blur")
args.add_argument("--scale", type=float, default=2, help="Scale factor to enlarge the blurred area")
args.add_argument("--confidence", type=float, default=0.5, help="Face detection confidence (0-1)")
args.add_argument("--model", type=int, default=1, choices=[0, 1], help="Face detection model: 0 for short-range, 1 for full-range")
args.add_argument("--oval-stretch", type=float, default=0.85, help="Stretch factor for oval blur width (less than 1 is narrower).")
args.add_argument("--persistence", type=int, default=5, help="Number of frames to keep blurring after face is lost.")

args = args.parse_args()


output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=args.model, min_detection_confidence=args.confidence) as face_detection:

    face_blurrer = FaceBlurrer(
        blur_amount=args.blur,
        blur_type=args.blur_type,
        feather=args.feather,
        scale=args.scale,
        oval_stretch=args.oval_stretch,
        persistence=args.persistence
    )

    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)

        img = face_blurrer.process_img(img, face_detection)

        # save image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:

            frame = face_blurrer.process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = face_blurrer.process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
        
        cap.release()
        cv2.destroyAllWindows()
