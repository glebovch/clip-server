import logging
from typing import List
from PIL import Image
from mosec import Server, Worker, get_logger
from mosec.errors import MosecError, ValidationError
from mosec.mixin import MsgpackMixin
import msgpack
from io import BytesIO
import gc
import torch
import clip
import os
from time import time
import json



logger = get_logger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

NUM_DEVICE = torch.cuda.device_count()
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", 1))


class Preprocess(MsgpackMixin, Worker):
    """Sample Preprocess worker"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: bytes) -> dict:
        logger.info(f"DATA keys: {data.keys()}")
        logger.info(f"DATA image type: {type(data['image'])}")
        try:
            image = Image.open(BytesIO(data["image"])).convert("RGB")
        except KeyError as err:
            raise Exception(f"cannot find key {err}") from err
        except Exception as err:
            raise Exception(f"cannot decode as image data: {err}") from err
        return {"image": image, "text": data["text"]}


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
        logger.info(f"using device {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)


    def forward(self, data: List[dict]) -> List[dict]:
        logger.info("Worker=%d on device=%s is processing...", self.worker_id, self.device)
        logger.info("Empty cache...")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            inference_time = time()
            image = torch.cat(tuple(self.preprocess(d["image"]).unsqueeze(0).to(self.device) for d in data))
            text = clip.tokenize([d["text"] for d in data]).to(self.device)
            logits_per_image, logits_per_text = self.model(image, text)
            inference_time = round(time() - inference_time, 2)
            result = {}
            result["similarities"] = logits_per_text.tolist()
            result["inference_time"] = inference_time
        except Exception as err:
            raise err

        return [{"similarities": s} for s in result["similarities"]]



class Postprocess(MsgpackMixin, Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()

    def forward(self, data: dict):
        return f"Similarities: {data['similarities']}"


def run_server():
    server = Server()
    server.append_worker(Preprocess)
    server.append_worker(Inference, max_batch_size=32)
    server.append_worker(Postprocess)
    server.run()


if __name__ == "__main__":
    run_server()

