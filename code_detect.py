import cv2
import numpy as np

# Configuration
image_path = "captured.jpg"
output_path = "processed.jpg"
url = "http://192.168.1.30:8080/video"  # IP Webcam URL

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Cannot access IP Webcam stream. Check network, IP address, and app status.")
    exit()

cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    output = frame.copy()
    H, W, _ = frame.shape

    # Parking spot layout parameters
    blocks_top = 2
    blocks_bottom = 2
    slots_per_block = 2
    slot_w, slot_h = 240, 300
    slot_spacing = 60
    vertical_spacing = 250
    left_offset = 200
    block_spacing = 80

    free_spots = []
    parking_spots = []
    slot_id = 1

    for row in range(2):
        current_blocks = blocks_top if row == 0 else blocks_bottom
        for block in range(current_blocks):
            total_block_width = slots_per_block * slot_w + (slots_per_block - 1) * slot_spacing
            block_x = left_offset + block * (total_block_width + block_spacing)
            block_y = (H - (2 * slot_h + vertical_spacing)) // 2 + row * (slot_h + vertical_spacing)

            for slot in range(slots_per_block):
                x = block_x + slot * (slot_w + slot_spacing)
                y = block_y

                parking_spots.append((slot_id, x, y, slot_w, slot_h, row, block))

                roi = frame[y:y + slot_h, x:x + slot_w]
                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                white_pixels = cv2.countNonZero(thresh)
                total_pixels = slot_w * slot_h

                if white_pixels / total_pixels < 0.9:
                    color = (0, 0, 255)  # Occupied - red
                else:
                    color = (0, 255, 0)  # Free - green
                    free_spots.append(slot_id)

                cv2.rectangle(output, (x, y), (x + slot_w, y + slot_h), color, 2)
                cv2.putText(output, f"#{slot_id}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                slot_id += 1

            block_width = slots_per_block * slot_w + (slots_per_block - 1) * slot_spacing
            cv2.rectangle(output,
                          (block_x - 10, block_y - 10),
                          (block_x + block_width + 10, block_y + slot_h + 10),
                          (255, 255, 0), 3)

            block_name = chr(65 + block + (row * blocks_top))
            cv2.putText(output, f"Block {block_name}", (block_x + 5, block_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def distance_to_entrance(x, y, w, h):
        center_x = x + w // 2
        center_y = y + h // 2
        entrance_x = W // 2
        entrance_y = H
        return np.hypot(center_x - entrance_x, center_y - entrance_y)

    if free_spots:
        free_slot_data = [
            (slot_id, x, y, w, h) for (slot_id, x, y, w, h, _, _) in parking_spots if slot_id in free_spots
        ]
        best_spot = min(free_slot_data, key=lambda s: distance_to_entrance(s[1], s[2], s[3], s[4]))

        slot_id, x, y, w, h = best_spot
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(output, "Best Spot", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Parking Detection", output)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(output_path, output)
        print(f"Image saved to: {output_path}")

cap.release()
cv2.destroyAllWindows()
