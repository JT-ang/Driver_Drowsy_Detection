import cv2


class Camera:
    """
    camera_origin: the source of the frame, such as : webcam, mp4
    """

    def __init__(self, camera_origin=0):
        self.origin = camera_origin

    def get_frames(self, output_path, frame_interval):
        """
        :param output_path: need save the frame, then save it in this path
        :param frame_interval:
        @details press 'q' to quit the camera
        """
        video = cv2.VideoCapture(self.origin)
        count = 0
        frame_num = 0
        extracted = []

        if not video.isOpened():
            print("Error opening video capture!")
            return

        # Read until the end of the video
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            count += 1

            if count % frame_interval == 0:
                frame_num += 1
                # frame_output_path = f"{output_path}/frame_{frame_num}.jpg"
                # cv2.imwrite(frame_output_path, frame)
                # cv2.imshow("frame", frame)
                extracted.append(frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        # Release the video object
        video.release()
        cv2.destroyAllWindows()

        return extracted


camera = Camera(0)

output_path = "extracted_frames"

# the interval of the extraction
frame_interval = 10

# Call the instance function to get the frames
res = camera.get_frames(output_path, frame_interval)
