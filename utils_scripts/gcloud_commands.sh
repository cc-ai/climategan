# WARNING this is a list of commands, do not execute as is, rather
# select the commands you want to use
# You need to download the `gcloud` cli first

# init and authentication with link to google.com
gcloud init --console-only

# authentication with json key downloaded from the cloud console
gcloud auth activate-service-account ~/Downloads/vicc-ai-951291074050.json




# create instance from snapshot
gcloud compute instances create vicc-instance \
    --zone=europe-west4-a \
    --machine-type=n1-highmem-96  \
    --image-family=torch-xla \
    --image-project=ml-images  \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --boot-disk-size=50GB \
    --source-snapshot=snapshot-vicc-xla

# -------------------------------------------------------------------------------
# START STANDARD WORKFLOW -------------------------------------------------------
# -------------------------------------------------------------------------------

# create instance from image
gcloud beta compute instances create vicc-instance \
    --zone=europe-west4-a \
    --machine-type=n1-highmem-96  \
    --source-machine-image=vicc-xla-image \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# create tpu
export VERSION=1.6
gcloud compute tpus create vicc-tpu \
    --zone=europe-west4-a \
    --network=default \
    --version=pytorch-${VERSION?}  \
    --accelerator-type=v3-8

# list created tpu instances
gcloud compute tpus list
#--zone=europe-west4-a

# ssh into instance
gcloud compute ssh vicc-instance
#--zone=europe-west4-a

# stop tpu
gcloud compute tpus stop vicc-tpu

# stop instance
gcloud compute instances stop vicc-instance

# -------------------------------------------------------------------------------
# END STANDARD WORKFLOW ---------------------------------------------------------
# -------------------------------------------------------------------------------

# delete tpu
gcloud compute tpus delete vicc-tpu

# delete instance
gcloud compute instances delete vicc-instance


# list VM instances
gcloud compute instances list #--zone=europe-west4-a


#/!\ when using gcloud compute scp, make sure the dir on the instance has the right permissions (typically run chmod -R 777)
# copy directory
gcloud compute scp --recurse "/miniscratch/schmidtv/ccai/omnigan/runs/painter/v0 (13)" vicc-instance:/home/victor/weights/v0/painter

# copy file
gcloud compute scp "/miniscratch/schmidtv/ccai/omnigan/runs/painter/v0 (13)/opts.yaml" vicc-instance:/home/victor/weights/v0/painter