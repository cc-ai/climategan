echo "Dowloading Script" && python download_labelbox.py

echo "Merging Script" && python merge_labelbox_masks.py

echo "Cleaning labeled"
rm /Users/victor/Downloads/metrics-v2/labels/*
cp /Users/victor/Downloads/labelbox_test_flood-v2/__labeled/* /Users/victor/Downloads/metrics-v2/labels

echo "Create labeled images Script" && python create_labeled.py