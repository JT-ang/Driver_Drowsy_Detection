import cv2
import os


class Camera:
    """
    camera_origin: the source of the frame, such as : webcam, mp4
    """

    def __init__(self, camera_origin=0):
        self.origin = camera_origin
        self.frame_buffer = []

    def get_frames(self, frame_interval, batch_size, show_images=False):
        """
        :param frame_interval:
        :param batch_size:
        @details press 'q' to quit the camera
        """
        video = cv2.VideoCapture(self.origin)
        count = 0
        frame_num = 0

        if not video.isOpened():
            print("Error opening video capture!")
            return

        # Read frames
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                frame_num += 1
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) >= batch_size:
                    break
        # Realise
        video.release()
        cv2.destroyAllWindows()

        return self.frame_buffer

    def save_frames(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        for i, frame in enumerate(self.frame_buffer):
            frame_output_path = os.path.join(folder_path, f"frame_{i + 1}.jpg")
            cv2.imwrite(frame_output_path, frame)

        print("Frames saved successfully!")

    def show_video(self):
        for frame in self.frame_buffer:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1000) & 0xFF == ord('n'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = Camera(0)
    frame_interval = 10
    batch_size = 2
    frames = camera.get_frames(frame_interval, batch_size)
    camera.show_video()
