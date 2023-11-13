> 选择多少帧抽一张图片，frame_count代表第多少帧
>
> ```python
>     if frame_count % 2 == 0:  # 每隔一帧保存图像
>         cv2.imwrite(f"extract/frame_{frame_count}.jpg", frame)
> ```

> 可以选择多少s暂停
>
> ```python
> # ed = time.time()
>     # print(ed - st)
>     # if ed - st >= 1:  # 1s后停止视频抽帧
>     #     break
> ```
>
> 
