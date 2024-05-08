from aiortc import MediaStreamTrack, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiohttp import web
import av
import cv2
import numpy as np
import asyncio
import time
def filter_frame(frame):
    # 将视频帧转换为NumPy数组
    img = frame.to_ndarray(format="bgr24")

    # 对图像进行滤波处理
    filtered = cv2.bilateralFilter(img, 9, 75, 75)

    return filtered


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print("已连接")

    # 创建一个设备对象，这里我们使用默认设备
    player = MediaPlayer('video=e2eSoft iVCam', format='dshow', options={
        'video_size': '640x480'
    })

    # 创建一个媒体接收器，这里我们将视频流保存到文件
    recorder = MediaRecorder("file.mp4")

    # 获取视频流
    video_track = player.video

    # 将视频流添加到媒体接收器
    recorder.addTrack(video_track)

    # 开始接收视频流
    await recorder.start()

    while True:
        msg = await ws.receive_str()
        if msg == 'start_counter':
            await ws.send_str('start_receiving')
            await asyncio.sleep(2)
            break

    i = 0
    while i <= 6:
        # 获取视频帧
        frame = await video_track.recv()

        # 记录开始时间
        start_time = time.time()

        # 滤波处理帧
        filtered_frame = filter_frame(frame)

        # 计算处理速率
        end_time = time.time()
        rate = 1.0 / (end_time - start_time)
        print(f"处理速率: {rate} 帧/秒")

        await ws.send_str(str(i))
        i += 1
        await asyncio.sleep(2.5)

    # 停止接收视频流
    await recorder.stop()

    print("已断开")
    return ws

app = web.Application()
app.router.add_route('GET', '/path', websocket_handler)
web.run_app(app, host='localhost', port=8080)