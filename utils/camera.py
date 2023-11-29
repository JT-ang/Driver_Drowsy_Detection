import cv2
import os


class Camera:
    """
    camera_origin: the source of the frame, such as : webcam, mp4
    """

    def __init__(self, logger, camera_origin=0):
        self.logger = logger
        self.frame_buffer = []
        self.origin = camera_origin
        self.video = cv2.VideoCapture(self.origin)
        self.logger.log_cli(f'Camera Inited!')

    def init_video(self):
        self.logger.log_cli('Video Heating!')
        count = 0
        while count < 60:
            ret, frame = self.video.read()
            if not ret:
                break
            count += 1
        self.logger.log_cli('Video Inited!')

    def get_frames(self, f_gap, bs):
        """
        :param f_gap:
        :param bs:
        @details press 'q' to quit the camera in show mode
        """
        count = 0
        frame_num = 0
        self.frame_buffer = []

        if not self.video.isOpened():
            self.logger.log_cli(f"Error Opening Video Capture!")
            return

        # Read frames
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            count += 1
            if count % f_gap == 0:
                frame_num += 1
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) >= bs:
                    break

        return self.frame_buffer

    def save_frames(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        for i, frame in enumerate(self.frame_buffer):
            frame_output_path = os.path.join(folder_path, f"frame_{i + 1}.jpg")
            cv2.imwrite(frame_output_path, frame)

        self.logger.log_cli(f"Frames Saved Successfully!")

    def show_video(self):
        for frame in self.frame_buffer:
            cv2.imshow("Video", frame)
            if cv2.waitKey(0) & 0xFF == ord('n'):
                break

        cv2.destroyAllWindows()

    def close(self):
        self.video.release()
        print('Camera Closed!')


# if __name__ == '__main__':
    # logger = Logger('./info.txt')
    # camera = Camera(logger)
    # camera.get_frames(1, 5)
    # # camera.save_frames('D:\\Data\\dataset\\drowsy')
    # camera.show_video()
