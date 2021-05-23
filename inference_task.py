# from celery.signals import worker_process_init
import os
import logging
import ssl

from celery import Celery, Task

from azure.storage.blob import BlobServiceClient
# from typing_extensions import TypeVarTuple
import torch

from omnigan.trainer import Trainer
from apply_celery_events import run_inference_from_trainer

try:
    app = Celery('inference_task', broker=os.environ.get('CELERY_BROKER_URL', 'amqp://admin:mypass@rabbitmq:5672'))
except:
    app = Celery('inference_task', broker='amqp://user:otiZlPbTy7@10.0.71.54:5672/')

broker_use_ssl = {
  'ca_certs': '/etc/ssl/certs/cacert.pem',
  'cert_reqs': ssl.CERT_REQUIRED
}
app.conf.update(
    broker_use_ssl=broker_use_ssl,
)

def connect_to_container(container):
    stringauth = os.environ.get('STORAGE_CONNECTION_STRING')
    return BlobServiceClient.from_connection_string(stringauth).get_container_client(container)

class OmniGAN(Task):

    def __init__(self):
        logging.info(f"Initializing the Trainer")
        device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # if XLA:
        #     device = xm.xla_device()
        resume_path = os.environ.get('RESUME_PATH', "model/masker/release-may-20")
        self._trainer = Trainer.resume_from_path(
            resume_path,
            setup=True,
            inference=True,
            new_exp=None,
            device=device,
        )
        self._trainer.G.half()
        self._container_client = connect_to_container('vicc')

    @property
    def trainer(self):
        return self._trainer

    @property
    def container(self):
        return self._container_client



@app.task(base=OmniGAN)
def infer(**kwargs):
    """This is the interface that the rabbitMQ task payload conforms to.
    Expected kwargs:
        paths_on_container [str]: list of paths to fetch the images from
        output_paths [str]: list of output paths to upload transformed images to
    """
    run_inference_from_trainer(infer.trainer, infer.container, **kwargs)
