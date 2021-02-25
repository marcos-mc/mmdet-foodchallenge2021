_base_ = './mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco.py'
# learning policy
lr_config = dict(step=[12])
runner = dict(max_epochs=20)
