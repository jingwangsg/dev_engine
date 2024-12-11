import json

from decord import VideoReader

from dev_engine.system import run_cmd


def probe_meta_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v error",
        "-select_streams v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,nb_frames,duration",
        "-of json",
    ]

    ret = run_cmd(" ".join(cmd), fault_tolerance=True)

    if ret.returncode != 0:
        print(f"Error probing video: {video_path}")
        return None

    probe_str = ret.stdout.strip("\n").strip()
    probe_dict = json.loads(probe_str)

    try:
        width = probe_dict["streams"][0]["width"]
        height = probe_dict["streams"][0]["height"]
        fps = probe_dict["streams"][0]["r_frame_rate"]
        duration = probe_dict["streams"][0]["duration"]
        num_frames = probe_dict["streams"][0]["nb_frames"]
    except Exception:
        print(f"Error parsing probe_strs: {probe_str}")
        return None

    def _fail_safe_convert(val, klass):
        try:
            return klass(val)
        except:
            return None

    width, height = _fail_safe_convert(width, int), _fail_safe_convert(height, int)
    fps = _fail_safe_convert(fps, eval)
    duration = _fail_safe_convert(duration, float)
    num_frames = _fail_safe_convert(num_frames, int)

    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames,
    }

    return meta


def probe_meta_decord(video_path):
    vr = VideoReader(video_path)
    wh = vr[0].shape[:2]

    length = len(vr)

    fps = vr.get_avg_fps()

    return {
        "width": wh[1],
        "height": wh[0],
        "fps": fps,
        "duration": length / fps,
        "num_frames": length,
    }


def probe_meta(video_path):
    try:
        return probe_meta_decord(video_path)
    except:
        return probe_meta_ffprobe(video_path)
