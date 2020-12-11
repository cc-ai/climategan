# from celery.signals import worker_process_init
import os
import logging
from celery import Celery, Task

# import torch

from omnigan.trainer import Trainer
from apply_celery_events import run_inference_from_trainer

app = Celery('inference_task', broker=os.environ.get('CELERY_BROKER_URL', 'amqp://admin:mypass@rabbitmq:5672'))

class OmniGAN(Task):

    def __init__(self):
        logging.info(f"Initializing the Trainer")
        # torch.multiprocessing.set_start_method('spawn')
        device = None
        # if XLA:
        #     device = xm.xla_device()
        resume_path = os.environ.get('RESUME_PATH', "tests-v1/model")
        self._trainer = Trainer.resume_from_path(
            resume_path,
            setup=True,
            inference=True,
            new_exp=None,
            input_shapes=(3, 640, 640),
            device=device,
        )

    @property
    def trainer(self):
        return self._trainer

@app.task(base=OmniGAN)
def infer(images=None, output=None):
    run_inference_from_trainer(infer.trainer, images_path='tests-v1/images', output_path='tests-v1/output', time_inference=True)
