from ultralytics import YOLO
import cv2
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    success, frame = cap.read()
    
    if success:
        results = model.predict(frame, conf=0.3, imgsz=1280)
        annotated_frame = results[0].plot()
        cv2.imshow("Apple Detection", annotated_frame)
        apples_detected = len(results[0].boxes)
        print(f"Detected {apples_detected} apples")
  
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() 
            conf = box.conf[0].item()  
            print(f"Apple at position: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), confidence: {conf:.2f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()